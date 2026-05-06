import torch
import torch.nn as nn

try:
    from .soft_thresholding import SoftThresh
except ImportError:
    from soft_thresholding import SoftThresh



class OneShotCFOCNN(nn.Module):


    def __init__(self, N_tau=128, device="cuda"):
        super().__init__()
        self.device = device
        self.N_tau = N_tau

        # 特征提取网络：处理复数数据的实部和虚部 (2通道)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 下采样
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 压缩空间维度
        ).to(device).to(torch.float64)

        # 回归器：预测 CFO 数值
        self.regressor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ).to(device).to(torch.float64)

    def forward(self, yf):
        # 1. 构造 CNN 输入 [Batch, 2, Ant, SC]
        x = torch.stack([yf.real, yf.imag], dim=1)

        # 2. 预测 CFO
        feat = self.cnn(x)
        feat = torch.flatten(feat, 1)
        cfo_est = self.regressor(feat)  # [Batch, 1]

        # 3. 构造补偿矩阵 e^(-j * 2*pi * delta_f * t)
        # 注意：这里假设最后一个维度是频率/时延轴 (N_tau)
        tau_grid = torch.arange(self.N_tau, device=self.device, dtype=torch.float64)
        phase_shift = -2 * torch.pi * cfo_est * tau_grid.reshape(1, 1, -1)
        cfo_comp_matrix = torch.exp(1j * phase_shift)

        # 4. 执行补偿
        yf_calibrated = yf * cfo_comp_matrix
        return yf_calibrated, cfo_est


# ==========================================
# 2. 硬件误差自适应校准层 (空间域)
# ==========================================
class ArrayCalibrationLayer(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0),
                              bias=False, dtype=torch.complex128).to(device)
        nn.init.dirac_(self.conv.weight)

    def forward(self, u):
        u_in = u.unsqueeze(1)
        u_calibrated = self.conv(u_in)
        return u_calibrated.squeeze(1)


# ==========================================
# 3. 深度展开 ADMM 的单层单元 (纯净版)
# ==========================================
class UnitCellCAdmmNet(nn.Module):
    """
    此单元不再处理 CFO，只负责在已对齐频率的基础上进行迭代信道估计。
    """

    def __init__(self, V_2d, beta=0.1, rho=1.0, device="cuda"):
        super().__init__()
        self.device = device
        self.S = SoftThresh(beta, device=device)
        self.V = torch.nn.Parameter(V_2d.clone().detach().to(device))

        # 修复了之前版本可能存在的类型转换警告
        self.rho = torch.nn.Parameter(
            torch.tensor([float(rho)], device=device, dtype=torch.float64)
        )

        # 空间校准层 (处理天线耦合)
        self.calibrator = ArrayCalibrationLayer(device)

    def forward(self, u_in, yf_clean):
        w = 1 / (self.V + self.rho).unsqueeze(0)

        # 1. 软阈值去噪
        s_out = self.S(u_in)

        # 2. 空间校准
        s_calibrated = self.calibrator(s_out)

        # 3. FFT 域极速更新
        fft_term = torch.fft.fft2(self.rho * (2 * s_calibrated - u_in) + yf_clean, dim=(1, 2))
        u_out = torch.fft.ifft2(w * fft_term, dim=(1, 2))

        # 4. 残差连接
        u_out = u_out + u_in - s_calibrated

        return u_out


# ==========================================
# 4. 主网络架构 (预补偿 + 迭代架构)
# ==========================================
class CAdmmNet(nn.Module):
    def __init__(self, V_2d, num_layers=15, beta=0.1, rho=1.0, device="cuda", N_tau=128):
        super().__init__()
        self.device = device
        self.num_layers = num_layers


        self.cfo_pre_estimator = OneShotCFOCNN(N_tau, device)


        self.layers = nn.ModuleList(
            [UnitCellCAdmmNet(V_2d, beta, rho, device) for _ in range(num_layers)]
        )
        self.S = SoftThresh(beta, device=device)

    def forward(self, yf, return_cfo=False):
        # 1. CNN 预处理：一次性消除频率偏置
        yf_clean, cfo_est = self.cfo_pre_estimator(yf)

        # 2. CADMM 迭代：在干净的信号上估计信道矩阵
        u = torch.zeros_like(yf_clean, device=self.device)
        for unit_cell in self.layers:
            u = unit_cell(u, yf_clean)

        # 3. 最终非线性激活
        u_final = self.S(u)

        if return_cfo:
            return u_final, cfo_est
        return u_final