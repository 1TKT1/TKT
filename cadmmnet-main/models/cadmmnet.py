import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftThresh(nn.Module):
    def __init__(self, beta, device="cuda"):
        super().__init__()
        self.device = device
        # 保持纯粹的学习标量 beta，后续在各层内部独立演进分化
        self.beta = torch.nn.Parameter(torch.tensor([float(beta)], device=device, dtype=torch.float32))

    def forward(self, x):
        mag = torch.sqrt(x.real ** 2 + x.imag ** 2 + 1e-12)
        threshold = torch.abs(self.beta)

        # 经典软阈值收缩算子
        scale = F.relu(mag - threshold) / (mag + 1e-12)
        return x * scale


class UnitCellCAdmmNet(nn.Module):
    def __init__(self, V_2d, beta=0.005, rho=1.0, device="cuda"):
        super().__init__()
        self.device = device
        self.S = SoftThresh(beta, device=device)
        self.V = torch.nn.Parameter(V_2d.clone().detach().to(device).to(torch.float32))
        self.rho = torch.nn.Parameter(torch.tensor([float(rho)], device=device, dtype=torch.float32))

        #  方案一核心：可学习的层间残差物理跳跃权重 (初始设为 0.1)
        self.gamma = torch.nn.Parameter(torch.tensor([0.1], device=device, dtype=torch.float32))

    def forward(self, u_in, x_dirty):
        denom = torch.abs(self.V) + torch.abs(self.rho) + 1e-6
        s_out = self.S(u_in)
        fft_term = torch.fft.fft2(torch.abs(self.rho) * (2 * s_out - u_in) + x_dirty, dim=(1, 2))
        u_out = torch.fft.ifft2(fft_term / denom.unsqueeze(0), dim=(1, 2))


        u_next = u_out + u_in - s_out


        return u_next + torch.tanh(self.gamma) * u_in


class CAdmmNet(nn.Module):
    def __init__(self, V_2d, A_theta, A_tau, num_layers=15, beta=0.005, rho=1.0, device="cuda"):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.register_buffer('A_theta_H', A_theta.T.conj().to(torch.complex64))
        self.register_buffer('A_tau_conj', A_tau.conj().to(torch.complex64))


        self.layers = nn.ModuleList([
            UnitCellCAdmmNet(
                V_2d,
                beta=beta * (1.5 - (i / num_layers)),
                rho=rho * (1.0 + (i / num_layers)),
                device=device
            ) for i in range(num_layers)
        ])

        self.S = SoftThresh(beta, device=device)

        M = self.A_theta_H.shape[1]
        K = self.A_tau_conj.shape[0]
        self.register_buffer('array_gain', torch.tensor((M * K) ** 0.5, dtype=torch.float32))

    def forward(self, yf):
        # 接收信号匹配增益抵消归一化
        yf_norm = yf / self.array_gain

        step1 = torch.einsum('nm, bmk -> bnk', self.A_theta_H, yf_norm)
        x_dirty = torch.einsum('bnk, kt -> bnt', step1, self.A_tau_conj)

        u = torch.zeros_like(x_dirty, device=self.device)
        for unit_cell in self.layers:
            u = unit_cell(u, x_dirty)

        u_final = self.S(u)
        return torch.sqrt(u_final.real ** 2 + u_final.imag ** 2 + 1e-12)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# #
# # 1. 纯净软
# #
# class SoftThresh(nn.Module):
#     def __init__(self, beta, device="cuda"):
#         super().__init__()
#         self.device = device
#         # 基础去噪门限，随层数深度独立演进，拒绝相对门限引发的马太效应
#         self.beta = torch.nn.Parameter(torch.tensor([float(beta)], device=device, dtype=torch.float32))
#
#     def forward(self, x):
#         mag = torch.sqrt(x.real ** 2 + x.imag ** 2 + 1e-12)
#         threshold = torch.abs(self.beta)
#
#         # 经典软阈值收缩算子
#         scale = F.relu(mag - threshold) / (mag + 1e-12)
#         return x * scale
#
# # ==============================================================================
# # 2. ADMM 深度展开细胞核 (免求逆 + 动态点扩散函数 + 残差跳跃)
# # ==============================================================================
# class UnitCellCAdmmNet(nn.Module):
#     def __init__(self, V_2d, beta=0.005, rho=1.0, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.S = SoftThresh(beta, device=device)
#
#         # 动态点扩散函数，吸纳网格内部的干涉失配
#         self.V = torch.nn.Parameter(V_2d.clone().detach().to(device).to(torch.float32))
#         self.rho = torch.nn.Parameter(torch.tensor([float(rho)], device=device, dtype=torch.float32))
#
#         # 物理残差流的层间跳跃权重，打通深层梯度，确保微弱多径存活
#         self.gamma = torch.nn.Parameter(torch.tensor([0.1], device=device, dtype=torch.float32))
#
#     def forward(self, u_in, x_dirty):
#         denom = torch.abs(self.V) + torch.abs(self.rho) + 1e-6
#         s_out = self.S(u_in)
#
#         # 核心算力引擎：2D-FFT 频域对角化免矩阵求逆
#         fft_term = torch.fft.fft2(torch.abs(self.rho) * (2 * s_out - u_in) + x_dirty, dim=(1, 2))
#         u_out = torch.fft.ifft2(fft_term / denom.unsqueeze(0), dim=(1, 2))
#
#         u_next = u_out + u_in - s_out
#
#         # 物理残差流，采用 tanh 锚定幅度防爆炸
#         return u_next + torch.tanh(self.gamma) * u_in
#
# # ==============================================================================
# # 3. 终极版 Taylor-CADMM-Net 主网络架构 (带恒模流形投影与严格量纲对齐)
# # ==============================================================================
# class CAdmmNet(nn.Module):
#     def __init__(self, V_2d, A_theta, A_tau, num_layers=15, beta=0.005, rho=1.0, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.num_layers = num_layers
#
#         # 数据驱动的泰勒离网字典
#         # 通过 .resolve_conj().clone() 彻底解决 Adam 优化器在 view_as_real 时的复数共轭延迟张量求导崩溃报错
#         self.A_theta_H = nn.Parameter(A_theta.T.conj().resolve_conj().clone().to(torch.complex64))
#         self.A_tau_conj = nn.Parameter(A_tau.conj().resolve_conj().clone().to(torch.complex64))
#
#         # 层级解耦演进（Layer-wise Parameterization）
#         self.layers = nn.ModuleList([
#             UnitCellCAdmmNet(
#                 V_2d,
#                 beta=beta * (1.5 - (i / num_layers)),
#                 rho=rho * (1.0 + (i / num_layers)),
#                 device=device
#             ) for i in range(num_layers)
#         ])
#
#         self.S = SoftThresh(beta, device=device)
#
#         M = self.A_theta_H.shape[1]
#         K = self.A_tau_conj.shape[0]
#
#
#         self.register_buffer('array_gain', torch.tensor(M * K, dtype=torch.float32))
#
#     def forward(self, yf):
#         # 1. 接收信号匹配增益物理归一化
#         yf_norm = yf / self.array_gain
#
#
#         A_theta_H_constrained = self.A_theta_H / (torch.abs(self.A_theta_H) + 1e-12)
#         A_tau_conj_constrained = self.A_tau_conj / (torch.abs(self.A_tau_conj) + 1e-12)
#
#         # 2. 生成自适应脏图 (Dirty Image)，使用受物理流形严格约束的弹性字典
#         step1 = torch.einsum('nm, bmk -> bnk', A_theta_H_constrained, yf_norm)
#         x_dirty = torch.einsum('bnk, kt -> bnt', step1, A_tau_conj_constrained)
#
#         # 3. 交替方向乘子法 (ADMM) 深度展开细胞级联迭代
#         u = torch.zeros_like(x_dirty, device=self.device)
#         for unit_cell in self.layers:
#             u = unit_cell(u, x_dirty)
#
#         # 4. 把关算子输出高分辨率、无能量泄露的稀疏谱图 (供后续测试脚本进行拉格朗日二级插值)
#         u_final = self.S(u)
#
#         # 5. 提取复数幅度作为网络最终输出
#         return torch.sqrt(u_final.real ** 2 + u_final.imag ** 2 + 1e-12)