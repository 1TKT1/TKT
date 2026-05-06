
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftThresh(nn.Module):
    def __init__(self, beta, device="cuda"):
        super().__init__()
        # 1. 全局基础门槛
        self.beta = torch.nn.Parameter(
            beta * torch.ones([1], device=device, dtype=torch.float64)
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor([0.0], device=device, dtype=torch.float64)
        )

        self.device = device

    def forward(self, x):
        mag = torch.abs(x).to(torch.float64)

        # --- 步骤 1：计算行与列的边缘能量
        # doa_energy: [Batch, N_theta, 1]
        doa_energy = torch.mean(mag, dim=2, keepdim=True)
        # tde_energy: [Batch, 1, N_tau]
        tde_energy = torch.mean(mag, dim=1, keepdim=True)

        joint_energy = doa_energy * tde_energy

        alpha_pos = F.relu(self.alpha)
        penalty = 1.0 / (1.0 + alpha_pos * joint_energy)


        dynamic_beta = self.beta * penalty

        # --- 步骤4：严格的相位无损近端收缩
        zeros = torch.zeros_like(mag)
        x_out = torch.exp(1j * x.angle()) * torch.max(mag - dynamic_beta, zeros)

        return x_out


# class SoftThresh(nn.Module):
#     def __init__(self, beta, device="cuda"):
#         super().__init__()
#         self.beta = torch.nn.Parameter(beta * torch.ones([1], device=device))
#         self.device = device
#
#     def forward(self, x):
#         zeros = torch.zeros(x.size(), device=self.device)
#         x = torch.exp(1j * x.angle()) * torch.max(
#             torch.abs(x) - self.beta, zeros
#         )
#         return x

