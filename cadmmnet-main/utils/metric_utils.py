import os
import torch
import h5py
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.initialization_utils import initialize_model, load_state

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def generate_gaussian_kernel_2d(kernel_size=7, sigma=0.5, device='cuda'):
    """
    生成2D高斯平滑核，用于外环对数拉格朗日插值前的谱图平滑
    """
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g).unsqueeze(0).unsqueeze(0)


def evaluate_model(model, dataset_test_path, num_layers, model_path=None, load_latest_state=False,
                   metric='rmse', bin_threshold=2, amp_threshold=0.4, device='cuda'):
    """
    完全体评估函数：内环原生尖锐谱图寻峰防漏检 + 外环对数拉格朗日连续域亚像素插值提精度
    """
    print(f"正在读取测试数据 (HDF5): {dataset_test_path}")

    with h5py.File(dataset_test_path, 'r') as f:
        total_samples = f['all_H_cfr']['real'].shape[0]
        is_snr_testset = 'snr_values' in f

        if is_snr_testset:
            num_test_vectors = total_samples
            obs_real = f['all_H_cfr']['real'][:]
            obs_imag = f['all_H_cfr']['imag'][:]
            labels = f['all_X_label'][:]
            snr_values = f['snr_values'][:].flatten()
            num_samples_per_snr = int(f['num_samples_per_snr'][0][0])
        else:
            num_training = int(total_samples * 0.8)
            num_test_vectors = total_samples - num_training
            obs_real = f['all_H_cfr']['real'][num_training:]
            obs_imag = f['all_H_cfr']['imag'][num_training:]
            labels = f['all_X_label'][num_training:]
            snr_values = [0]
            num_samples_per_snr = num_test_vectors

        measurement_vectors = torch.complex(
            torch.from_numpy(obs_real).to(torch.float32),
            torch.from_numpy(obs_imag).to(torch.float32)
        ).transpose(1, 2).to(device)

        ground_truth = torch.from_numpy(labels).to(torch.float32).transpose(1, 2).to(device)

        if 'theta_grid' in f and 'tau_grid' in f:
            grid_theta = torch.from_numpy(f['theta_grid'][:]).squeeze().cpu().numpy()
            grid_tau = torch.from_numpy(f['tau_grid'][:]).squeeze().cpu().numpy()
        else:
            N_a, N_d = 128, 128
            sin_theta = np.linspace(np.sin(np.deg2rad(-60)), np.sin(np.deg2rad(60)), N_a)
            grid_theta = np.rad2deg(np.arcsin(sin_theta))
            grid_tau = np.linspace(10e-9, 200e-9, N_d)

    N_a = len(grid_theta)
    N_d = len(grid_tau)
    array_type = "2d_ofdm"

    # 初始化模型并加载最新的物理约束权重
    network = initialize_model(model, dataset_test_path, num_layers, device)
    model_tag = load_state(network, None, array_type=array_type, load_latest_state=load_latest_state,
                           model_path=model_path, return_tag=True)
    network.eval()

    # 准备平滑核（用于插值分支）
    kernel_2d = generate_gaussian_kernel_2d(7, 0.5, device)
    padding = 7 // 2

    print(f"\n--- 🚀 开始联合评估: {model_tag} (双域解耦: 尖锐域防漏检 + 平滑域超分辨) ---")

    with torch.no_grad():
        for snr_idx, snr in enumerate(snr_values):
            start_idx = snr_idx * num_samples_per_snr
            end_idx = start_idx + num_samples_per_snr

            batch_mv = measurement_vectors[start_idx:end_idx]
            batch_gt = ground_truth[start_idx:end_idx]

            # 网络前向传播输出
            batch_spec_raw = network(batch_mv)

            # 为插值分支专门做 2D 圆周填充与高斯平滑
            batch_spec_pad = F.pad(batch_spec_raw.unsqueeze(1), (padding, padding, padding, padding), mode='circular')
            batch_spec_blurred = F.conv2d(batch_spec_pad, kernel_2d).squeeze(1)

            doa_sq_errors = []
            tde_sq_errors = []
            total_targets = 0
            detected_targets = 0

            for i in range(batch_spec_raw.shape[0]):
                gt_img = batch_gt[i]

                # 核心机制：解耦双图策略
                # 1. 用来寻找离散坐标和执行擦除的，是没有光晕的“原生尖锐图”
                est_raw_img = batch_spec_raw[i].clone()
                # 2. 用来计算连续域抛物线拟合能量的，是有光晕的“高斯平滑图”
                est_blur_img = batch_spec_blurred[i]

                true_coords = torch.nonzero(gt_img > 1e-4)
                L_true = true_coords.shape[0]
                if L_true == 0:
                    continue

                total_targets += L_true
                est_coords_list = []
                est_offsets_list = []

                # ======================================================================
                # 🛠️ 核心修复点：完美抑制半径（Sufficient Suppress Radius）
                # 动态可学习字典让谱峰具有了一定的“物理弹性宽度”，不再是孤立的单像素点。
                # 必须将其设为 2（即 5x5 的邻域块擦除），才能干净彻底地抹去谱峰边缘残留，
                # 强迫下一次循环的 argmax 走向真正的其他弱径，一举解放检测率！
                # ======================================================================
                suppress_radius = 2

                for _ in range(L_true):
                    # 在尖锐谱图上捕获当前全局最大能量点的离散网格索引
                    max_idx = torch.argmax(est_raw_img)
                    r = int((max_idx // N_d).item())
                    c = int((max_idx % N_d).item())

                    # 在高斯平滑谱图上提取邻域能量，进行对数抛物线拟合（游标卡尺读数）
                    p_0 = torch.log(est_blur_img[r, c] + 1e-12)
                    p_r_minus = torch.log(est_blur_img[(r - 1) % N_a, c] + 1e-12)
                    p_r_plus = torch.log(est_blur_img[(r + 1) % N_a, c] + 1e-12)
                    p_c_minus = torch.log(est_blur_img[r, (c - 1) % N_d] + 1e-12)
                    p_c_plus = torch.log(est_blur_img[r, (c + 1) % N_d] + 1e-12)

                    # 二阶导数解析倒推物理偏移量
                    dr_offset = 0.5 * (p_r_minus - p_r_plus) / (p_r_minus - 2 * p_0 + p_r_plus + 1e-12)
                    dc_offset = 0.5 * (p_c_minus - p_c_plus) / (p_c_minus - 2 * p_0 + p_c_plus + 1e-12)

                    # 严格限制半网格约束，防止插值坐标跨界漂移
                    dr_offset = torch.clamp(dr_offset, -0.5, 0.5).item()
                    dc_offset = torch.clamp(dc_offset, -0.5, 0.5).item()

                    est_coords_list.append([r, c])
                    est_offsets_list.append([dr_offset, dc_offset])

                    # 核心改进：在尖锐图上进行 $5 \times 5$ 的彻底微距点位擦除，以绝后患
                    for dr in range(-suppress_radius, suppress_radius + 1):
                        for dc in range(-suppress_radius, suppress_radius + 1):
                            est_raw_img[(r + dr) % N_a, (c + dc) % N_d] = 0.0

                est_coords = torch.tensor(est_coords_list, dtype=torch.float32, device=device)
                est_offsets = torch.tensor(est_offsets_list, dtype=torch.float32, device=device)

                # 融合成连续域的超分辨率真实坐标
                est_coords_float = est_coords + est_offsets

                # 计算真实点与连续估计点之间的双向距离矩阵（考虑圆周阵列边界边界效应）
                t_r = true_coords[:, 0].unsqueeze(1)
                e_r = est_coords_float[:, 0].unsqueeze(0)
                dr = torch.abs(t_r - e_r)
                dr_circular = torch.minimum(dr, N_a - dr).float()

                t_c = true_coords[:, 1].unsqueeze(1)
                e_c = est_coords_float[:, 1].unsqueeze(0)
                dc = torch.abs(t_c - e_c)
                dc_circular = torch.minimum(dc, N_d - dc).float()

                cost_matrix = torch.sqrt(dr_circular ** 2 + dc_circular ** 2).cpu().numpy()

                # 匈牙利算法（二分图最优匹配），绝对不允许对同一个多径进行多重计数
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # 网格误差判定范围：4个物理网格间距以内视为有效召回
                valid_mask = cost_matrix[row_ind, col_ind] <= 4.0
                valid_rows = row_ind[valid_mask]
                valid_cols = col_ind[valid_mask]

                detected_targets += np.sum(valid_mask)

                if len(valid_rows) > 0:
                    true_theta = grid_theta[true_coords[valid_rows, 0].cpu().numpy()]
                    true_tau = grid_tau[true_coords[valid_rows, 1].cpu().numpy()]

                    # 将角度/时延网格物理刻度解包并映射回连续变化的真实物理单位 (度 / 秒)
                    est_r_cont = est_coords_float[valid_cols, 0].cpu().numpy()
                    delta_r = est_r_cont - true_coords[valid_rows, 0].cpu().numpy()
                    adj_est_r = true_coords[valid_rows, 0].cpu().numpy() + np.where(
                        delta_r > N_a // 2, delta_r - N_a, np.where(delta_r < -N_a // 2, delta_r + N_a, delta_r)
                    )

                    est_c_cont = est_coords_float[valid_cols, 1].cpu().numpy()
                    delta_c = est_c_cont - true_coords[valid_rows, 1].cpu().numpy()
                    adj_est_c = true_coords[valid_rows, 1].cpu().numpy() + np.where(
                        delta_c > N_d // 2, delta_c - N_d, np.where(delta_c < -N_d // 2, delta_c + N_d, delta_c)
                    )

                    # 子网格物理刻度双线性解析插值器
                    def interp_grid(grid_array, float_indices, max_len):
                        floor_idx = np.floor(float_indices).astype(int)
                        ceil_idx = floor_idx + 1
                        weight = float_indices - floor_idx
                        floor_idx = np.clip(floor_idx, 0, max_len - 1)
                        ceil_idx = np.clip(ceil_idx, 0, max_len - 1)
                        return grid_array[floor_idx] * (1 - weight) + grid_array[ceil_idx] * weight

                    est_theta = interp_grid(grid_theta, adj_est_r, N_a)
                    est_tau = interp_grid(grid_tau, adj_est_c, N_d)

                    # 累计均方误差平方和
                    doa_err = (true_theta - est_theta) ** 2
                    tde_err = (true_tau - est_tau) ** 2

                    doa_sq_errors.extend(doa_err.tolist())
                    tde_sq_errors.extend(tde_err.tolist())

            # 汇总当前SNR水平下的最终学术评估三项硬指标
            pd = (detected_targets / total_targets) * 100 if total_targets > 0 else 0
            rmse_doa = np.sqrt(np.mean(doa_sq_errors)) if len(doa_sq_errors) > 0 else 0
            rmse_tde = np.sqrt(np.mean(tde_sq_errors)) if len(tde_sq_errors) > 0 else 0
            rmse_tde_ns = rmse_tde * 1e9

            print(
                f" SNR = {int(snr):2d} dB |  成功检测率: {pd:5.1f}% |  极限 DOA RMSE: {rmse_doa:5.2f} ° | 极限 TDE RMSE: {rmse_tde_ns:5.2f} ns")