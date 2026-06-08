import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftThresh(nn.Module):
    """自适应 2D 联合能量软阈值层 """

    def __init__(self, beta, device="cuda"):
        super().__init__()
        self.device = device
        self.beta = torch.nn.Parameter(
            torch.tensor([float(beta)], device=device, dtype=torch.float32)
        )
        self.alpha = torch.nn.Parameter(
            torch.tensor([0.0], device=device, dtype=torch.float32)
        )

    def forward(self, x):
        mag = torch.sqrt(x.real ** 2 + x.imag ** 2 + 1e-12)

        # 联合稀疏性特征提取 (完美保留你的 2D 物理先验)
        doa_energy = torch.mean(mag, dim=2, keepdim=True)
        tde_energy = torch.mean(mag, dim=1, keepdim=True)
        joint_energy = doa_energy * tde_energy


        alpha_pos = torch.abs(self.alpha)
        beta_pos = torch.abs(self.beta)

        # 动态软阈值过滤
        # 当某处存在联合高能量时，penalty 会小于 1，从而自适应降低该处的截断门限，保护弱多径信号！
        penalty = 1 / (1 + alpha_pos * joint_energy)
        threshold = beta_pos * penalty

        scale = F.relu(mag - threshold) / (mag + 1e-12)
        return x * scale