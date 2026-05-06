# # # import torch
# # # import torch.nn as nn
# # #
# # # # 【核心修复】：处理同一文件夹下的模块导入问题
# # # try:
# # #     from .soft_thresholding import SoftThresh
# # # except ImportError:
# # #     # 兼容某些直接从当前目录运行脚本的情况
# # #     from soft_thresholding import SoftThresh
# # #
# # # class UnitCellCAdmmNet(nn.Module):
# # #     def __init__(self, V_2d, beta=0.1, rho=1.0, device="cuda"):
# # #         super().__init__()
# # #         self.device = device
# # #         self.S = SoftThresh(beta, device=device)
# # #
# # #         # V_2d 是预计算的二维频域特征矩阵
# # #         self.V = torch.nn.Parameter(V_2d.clone().detach().to(device))
# # #         self.rho = torch.nn.Parameter(
# # #             rho * torch.ones([1], device=device, dtype=torch.float64)
# # #         )
# # #
# # #     def forward(self, u_in, yf):
# # #         # 1. 计算求逆项的标量特征值
# # #         w = 1 / (self.V + self.rho).unsqueeze(0)
# # #
# # #         # 2. 空频域变换，注意必须在 dim=(1,2) 上做 2D-FFT，避开 Batch 维度
# # #         fft_term = torch.fft.fft2(self.rho * (2 * self.S(u_in) - u_in) + yf, dim=(1, 2))
# # #
# # #         # 3. 频域标量相乘并 IFFT 逆变换回空间域
# # #         u_out = torch.fft.ifft2(w * fft_term, dim=(1, 2))
# # #
# # #         # 4. 残差跳跃连接
# # #         u_out = u_out + u_in - self.S(u_in)
# # #         return u_out
# # #
# # #
# # # class CAdmmNet(nn.Module):
# # #     def __init__(self, V_2d, num_layers=15, beta=0.1, rho=1.0, device="cuda"):
# # #         super().__init__()
# # #         self.device = device
# # #         self.S = SoftThresh(beta, device=device)
# # #         self.num_layers = num_layers
# # #
# # #         self.layers = nn.ModuleList(
# # #             [UnitCellCAdmmNet(V_2d, beta, rho, device) for _ in range(num_layers)]
# # #         )
# # #
# # #     def forward(self, yf):
# # #         # 初始化与 yf 同维度的全零张量
# # #         u = torch.zeros_like(yf, device=self.device)
# # #
# # #         for unit_cell in self.layers:
# # #             u = unit_cell(u, yf)
# # #
# # #         return self.S(u)
# # import torch
# # import torch.nn as nn
# #
# # try:
# #     from .soft_thresholding import SoftThresh
# # except ImportError:
# #     from soft_thresholding import SoftThresh
# #
# #
# # class UnitCellCAdmmNet(nn.Module):
# #     def __init__(self, V_2d, beta=0.1, rho=1.0, device="cuda"):
# #         super().__init__()
# #         self.device = device
# #
# #
# #         self.S = SoftThresh(beta, device=device)
# #
# #         self.V = torch.nn.Parameter(V_2d.clone().detach().to(device))
# #         self.rho = torch.nn.Parameter(
# #             rho * torch.ones([1], device=device, dtype=torch.float64)
# #         )
# #
# #     def forward(self, u_in, yf):
# #         w = 1 / (self.V + self.rho).unsqueeze(0)
# #         fft_term = torch.fft.fft2(self.rho * (2 * self.S(u_in) - u_in) + yf, dim=(1, 2))
# #         u_out = torch.fft.ifft2(w * fft_term, dim=(1, 2))
# #         u_out = u_out + u_in - self.S(u_in)
# #         return u_out
# #
# #
# # class CAdmmNet(nn.Module):
# #     def __init__(self, V_2d, num_layers=15, beta=0.1, rho=1.0, device="cuda"):
# #         super().__init__()
# #         self.device = device
# #         self.num_layers = num_layers
# #
# #         # 【修复点 2】：输出层同样改回极简调用方式
# #         self.S = SoftThresh(beta, device=device)
# #
# #         self.layers = nn.ModuleList(
# #             [UnitCellCAdmmNet(V_2d, beta, rho, device) for _ in range(num_layers)]
# #         )
# #
# #     def forward(self, yf):
# #         u = torch.zeros_like(yf, device=self.device)
# #         for unit_cell in self.layers:
# #             u = unit_cell(u, yf)
# #         return self.S(u)
# import torch
# import torch.nn as nn
#
# try:
#     from .soft_thresholding import SoftThresh
# except ImportError:
#     from soft_thresholding import SoftThresh
#
#
# # ==========================================
# # 硬件误差自适应校准层
# # ==========================================
# class ArrayCalibrationLayer(nn.Module):
#     def __init__(self, device="cuda"):
#         super().__init__()
#         self.conv = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0),
#                               bias=False, dtype=torch.complex128).to(device)
#
#
#         nn.init.dirac_(self.conv.weight)
#
#     def forward(self, u):
#         # u: [Batch, N_theta, N_tau]
#         u_in = u.unsqueeze(1)
#         u_calibrated = self.conv(u_in)
#         return u_calibrated.squeeze(1)
#
#
# # ==========================================
# # 深度展开 ADMM 的单层单元
# # ==========================================
# class UnitCellCAdmmNet(nn.Module):
#     def __init__(self, V_2d, beta=0.1, rho=1.0, device="cuda"):
#         super().__init__()
#         self.device = device
#
#         self.S = SoftThresh(beta, device=device)
#         self.V = torch.nn.Parameter(V_2d.clone().detach().to(device))
#         self.rho = torch.nn.Parameter(
#             rho * torch.ones([1], device=device, dtype=torch.float64)
#         )
#
#         #  插入校准层
#         self.calibrator = ArrayCalibrationLayer(device)
#
#     def forward(self, u_in, yf):
#         w = 1 / (self.V + self.rho).unsqueeze(0)
#
#         # 核心 1：软阈值去噪与能量聚焦
#         s_out = self.S(u_in)
#
#         # 核心 2：让校准层去抵消互耦合带来的旁瓣不对称畸变
#         s_calibrated = self.calibrator(s_out)
#
#         # 核心 3：FFT 域极速求逆
#         fft_term = torch.fft.fft2(self.rho * (2 * s_calibrated - u_in) + yf, dim=(1, 2))
#         u_out = torch.fft.ifft2(w * fft_term, dim=(1, 2))
#
#         # 核心 4：残差更新
#         u_out = u_out + u_in - s_calibrated
#
#         return u_out
#
#
# # ==========================================
# # 主网络架构
# # ==========================================
# class CAdmmNet(nn.Module):
#     def __init__(self, V_2d, num_layers=15, beta=0.1, rho=1.0, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.num_layers = num_layers
#
#         self.S = SoftThresh(beta, device=device)
#
#         self.layers = nn.ModuleList(
#             [UnitCellCAdmmNet(V_2d, beta, rho, device) for _ in range(num_layers)]
#         )
#
#     def forward(self, yf):
#         u = torch.zeros_like(yf, device=self.device)
#         for unit_cell in self.layers:
#             u = unit_cell(u, yf)
#         return self.S(u)

