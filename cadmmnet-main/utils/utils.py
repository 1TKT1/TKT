import random
import math
from datetime import datetime
import torch
import torch.nn.functional as F


def f2angle(f):
    f = f - 1 / 2
    f_clamped = torch.clamp(2 * f, -1.0, 1.0)
    theta = torch.asin(f_clamped) + torch.pi / 2
    return torch.rad2deg(theta)


def nmse(X1, X2):
    d = torch.norm(X1 - X2) ** 2
    n = torch.norm(X2) ** 2
    r = torch.mean(d / (n + 1e-12))
    return r


def set_random_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def model_id(network):
    model_name = network.__class__.__name__.lower()
    num_layers = getattr(network, 'num_layers', 15)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_tag = f"{model_name}_{num_layers}l_{timestamp}"
    return model_name, model_tag


def toeplitz_to_vector(X):
    return X


def circulant_hermitian_to_vector(X):
    return X



def k_largest_peaks_2d(spectrum_flat, N_angle, N_delay, k=15):
    """
    在展平的 2D 空间-时延谱中，通过 2D MaxPool 提取局部极大值点，并返回前 k 个最强峰值的展平索引。
    """
    device = spectrum_flat.device
    # 1. 还原为 2D 图像结构 [1, 1, N_angle, N_delay]
    spec_2d = spectrum_flat.view(1, 1, N_angle, N_delay)

    # 2. 局部最大值滤波 (3x3 邻域非极大值抑制)
    pooled = F.max_pool2d(spec_2d, kernel_size=3, stride=1, padding=1)

    # 3. 寻找峰值掩膜：既是局部最大值，又必须大于基础噪底
    peaks_mask = (spec_2d == pooled) & (spec_2d > 0.01 * torch.max(spec_2d))
    peaks_mask_flat = peaks_mask.view(-1)

    # 4. 过滤出所有候选峰值的物理索引和对应的能量幅值
    candidate_indices = torch.nonzero(peaks_mask_flat).squeeze(-1)
    if candidate_indices.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    candidate_amps = spectrum_flat[candidate_indices]

    # 5. 排序并精准截取前 k 个能量最强的多径峰值
    actual_k = min(k, candidate_indices.numel())
    _, topk_meta_idx = torch.topk(candidate_amps, actual_k, descending=True)

    return candidate_indices[topk_meta_idx]