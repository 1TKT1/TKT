import random
import math  # 【新增】引入 math 库用于处理 nan 和开方运算
from datetime import datetime

import torch
import torch.nn.functional as F


def f2angle(f):
    """
    将频率 f (范围 [-0.5, 0.5]) 转换为对应的物理角度 (范围 [-90°, 90°]
    """
    f = f - 1 / 2
    # 限制范围以防止因数值误差导致 arcsin 报错 NaN
    f_clamped = torch.clamp(2 * f, -1.0, 1.0)
    theta = torch.asin(f_clamped) + torch.pi / 2
    return torch.rad2deg(theta)


def nmse(X1, X2):
    """
    计算两个张量之间的归一化均方误差 (Normalized Mean Squared Error)
    """
    d = torch.norm(X1 - X2) ** 2
    n = torch.norm(X2) ** 2
    # 防止分母为 0
    r = torch.mean(d / (n + 1e-12))
    return r


def db(x: torch.Tensor):
    """
    将幅度平方值转换为分贝 (dB)
    """
    # 加上极小值防止 log10(0) 得到 -inf
    return 10 * torch.log10(torch.abs(x) + 1e-12)


def set_random_seeds(seed: int):
    """
    设置全局随机种子以确保实验可复现
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def model_id(network):
    """
    基于模型名称、层数和当前时间戳生成唯一的模型标识符 (Tag)
    """
    model_name = network.__class__.__name__.lower()
    num_layers = network.num_layers
    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_tag = f"{model_name}_{num_layers}l_{timestamp}"
    return model_name, model_tag


def randu(shape, a, b):
    """
    生成区间 [a, b] 上均匀分布的随机张量
    """
    random_tensor = a + (b - a) * torch.rand(shape)
    return random_tensor


def randn(shape, mean, std_dev):
    """
    生成均值为 mean，标准差为 std_dev 的正态分布随机张量
    """
    random_tensor = mean + std_dev * torch.randn(shape)
    return random_tensor


def randi(low, high, k):
    """
    在区间 [low, high] (包含两端) 内无放回地随机抽取 k 个整数
    """
    assert k <= (high - low + 1), "k cannot be larger than the number of available integers."
    all_integers = torch.arange(low, high + 1)
    random_selection = all_integers[torch.randperm(all_integers.size(0))][:k]
    return random_selection


def idxf(freqs, frequency_grid):
    """
    在 1D 频率网格中寻找距离目标频率最近的网格点索引 (用于构建 Ground Truth)
    """
    K = freqs.size(0)
    N = frequency_grid.size(0)
    closest_indices = torch.zeros(K, dtype=torch.long)
    for i in range(K):
        pos = torch.nonzero(frequency_grid >= freqs[i], as_tuple=True)[0]
        if pos.numel() == 0:
            closest_indices[i] = N - 1
        elif pos[0] == 0:
            closest_indices[i] = 0
        else:
            pos = pos[0]
            if torch.abs(frequency_grid[pos] - freqs[i]) < torch.abs(frequency_grid[pos - 1] - freqs[i]):
                closest_indices[i] = pos
            else:
                closest_indices[i] = pos - 1
    return torch.sort(closest_indices)[0]


def k_largest_peaks_2d(spectrum_1d, N_angle, N_delay, k):
    """
    【新增函数】：2D 联合网格峰值搜索
    在展平的二维伪谱中寻找目标的联合 TDE 和 DOA 坐标。

    参数:
        spectrum_1d: 展平的一维频谱张量 (大小为 N_angle * N_delay)
        N_angle: 角度维度的网格数量
        N_delay: 时延维度的网格数量
        k: 需要寻找的最大峰值数量

    返回:
        top_k_indices_1d: 展平后的峰值 1D 索引张量，按幅度从大到小排列
    """
    # 1. 确保输入为幅度谱，并将其重塑为 2D 图像格式 [1, 1, H, W]
    x_2d = spectrum_1d.view(N_angle, N_delay).abs()
    x_tensor = x_2d.unsqueeze(0).unsqueeze(0)

    # 2. 3x3 邻域最大值池化，步长为1，通过 padding=1 保持原始维度
    pooled = F.max_pool2d(x_tensor, kernel_size=3, stride=1, padding=1)

    # 3. 提取局部最大值点（当且仅当该点大于极小阈值防止背景噪声，且等于邻域内的最大值）
    is_peak = (x_tensor == pooled) & (x_tensor > 1e-6)
    is_peak = is_peak.squeeze()

    # 4. 获取峰值幅度和二维坐标
    peak_values = x_2d[is_peak]
    peak_indices = torch.nonzero(is_peak)  # 形状: [num_peaks, 2]

    # 安全检查：如果图中没有任何满足条件的峰值，直接返回空张量
    num_peaks_found = len(peak_values)
    if num_peaks_found == 0:
        return torch.tensor([], dtype=torch.long, device=spectrum_1d.device)

    # 5. 获取幅度最大的前 k 个峰值
    k_actual = min(k, num_peaks_found)
    _, sorted_idx = torch.sort(peak_values, descending=True)
    top_k_indices_2d = peak_indices[sorted_idx[:k_actual]]

    # 6. 将找到的二维峰值坐标 (a_idx, d_idx) 转换回 1D 展平索引，适配外部评测
    top_k_indices_1d = top_k_indices_2d[:, 0] * N_delay + top_k_indices_2d[:, 1]

    return top_k_indices_1d



def calculate_physical_errors(spectrum_1d, true_doa_cont, true_tde_cont, grid_config):

    N_angle = grid_config['N_angle']
    N_delay = grid_config['N_delay']

    # 1. 获取网络预测的最强峰值 1D 索引
    pred_indices_1d = k_largest_peaks_2d(spectrum_1d, N_angle, N_delay, k=1)

    if len(pred_indices_1d) == 0:
        # 如果网络没预测出任何峰值（全零张量等异常情况），返回 NaN 丢弃该样本
        return float('nan'), float('nan')

    best_idx = pred_indices_1d[0]

    # 2. 将 1D 索引还原为 2D 网格坐标
    # 注意：这里的整除和取余逻辑必须和 spectrum_1d.view(N_angle, N_delay) 保持一致
    pred_a_idx = best_idx // N_delay
    pred_d_idx = best_idx % N_delay

    # 3. 将网格坐标映射回真实的物理单位
    # 这里分母用 (N - 1) 是假设网格点包含了边界值 (类似 np.linspace)
    angle_step = (grid_config['angle_max'] - grid_config['angle_min']) / (N_angle - 1)
    delay_step = (grid_config['delay_max'] - grid_config['delay_min']) / (N_delay - 1)

    pred_physical_doa = grid_config['angle_min'] + pred_a_idx * angle_step
    pred_physical_tde = grid_config['delay_min'] + pred_d_idx * delay_step

    # 4. 计算并返回与真实连续物理量的平方误差 (Squared Error)
    doa_se = (pred_physical_doa.item() - true_doa_cont) ** 2
    tde_se = (pred_physical_tde.item() - true_tde_cont) ** 2

    return doa_se, tde_se