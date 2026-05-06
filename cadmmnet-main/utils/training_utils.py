import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 导入自定义工具
try:
    from .initialization_utils import initialize_model, load_state
    from .utils import model_id
except ImportError:
    from initialization_utils import initialize_model, load_state
    from utils import model_id

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATES_PATH = os.path.join(BASE_PATH, "models", "states")
LOSSES_PATH = os.path.join(BASE_PATH, "outputs", "losses")


# ==========================================
# 1. Dataset 定义 (对接新版 MATLAB 标签 & 强制 Float32)
# ==========================================
class CSI_Dataset(Dataset):
    def __init__(self, mat_path, indices=None):
        self.mat_path = mat_path
        self.file = None
        self.indices = indices

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        with h5py.File(self.mat_path, 'r') as f:
            # 获取样本总数，根据 MATLAB 的 -v7.3 保存格式，最后/最前一维通常是样本数
            return f['all_H_obs']['real'].shape[0]

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.mat_path, 'r')

        actual_idx = self.indices[idx] if self.indices is not None else idx

        # 1. 读取 CSI 数据 (HDF5 读取出来默认可能是 float64)
        obs_real = self.file['all_H_obs']['real'][actual_idx]
        obs_imag = self.file['all_H_obs']['imag'][actual_idx]
        label_real = self.file['all_H_label']['real'][actual_idx]
        label_imag = self.file['all_H_label']['imag'][actual_idx]

        # 2. 转换为复数张量，并【强制转换为 complex64 (即 float32 的复数版)】防止报错
        yf = torch.complex(
            torch.from_numpy(obs_real).to(torch.float32),
            torch.from_numpy(obs_imag).to(torch.float32)
        )
        x_gt = torch.complex(
            torch.from_numpy(label_real).to(torch.float32),
            torch.from_numpy(label_imag).to(torch.float32)
        )

        # 3. 将天线与符号维度处理为模型所需的 [16, 432]
        # 根据你之前数据的维度，这里取第一个符号并重塑
        if yf.dim() == 4:  # 如果维度是 [Tx, Rx, Sym, SC]
            yf = yf[:, :, 0, :].reshape(-1, yf.shape[-1])
            x_gt = x_gt[:, :, 0, :].reshape(-1, x_gt.shape[-1])
        elif yf.dim() == 3:  # 如果已经是 [Antennas, Sym, SC]
            yf = yf[:, 0, :].reshape(-1, yf.shape[-1])
            x_gt = x_gt[:, 0, :].reshape(-1, x_gt.shape[-1])

        # 4. 读取物理真实值标签，并【强制转换为 float32】
        try:
            cfo_gt = float(self.file['all_CFO_label'][actual_idx])
            doa_gt = float(self.file['all_DOA_label'][actual_idx])
            tde_gt = float(self.file['all_TDE_label'][actual_idx])
        except KeyError:
            cfo_gt, doa_gt, tde_gt = 0.0, 0.0, 0.0

        # 返回观测数据、标签数据、物理参数字典
        return yf, x_gt, {
            'cfo': torch.tensor(cfo_gt, dtype=torch.float32),
            'doa': torch.tensor(doa_gt, dtype=torch.float32),
            'tde': torch.tensor(tde_gt, dtype=torch.float32)
        }


# ==========================================
# 2. 数据加载设置
# ==========================================
def training_setup(mat_dataset_path, num_training_samples=None, batch_size=32):
    if not os.path.exists(mat_dataset_path):
        raise FileNotFoundError(f"未找到数据集文件: {mat_dataset_path}")

    with h5py.File(mat_dataset_path, 'r') as f:
        total_samples = f['all_H_obs']['real'].shape[0]

    if num_training_samples is not None and num_training_samples < total_samples:
        num_training = int(num_training_samples)
    else:
        num_training = int(total_samples * 0.8)

    num_validation = total_samples - num_training
    print(f"数据索引划分完毕 [总数: {total_samples} | 训练集: {num_training} | 验证集: {num_validation}]")

    indices = list(range(total_samples))
    train_indices = indices[:num_training]
    val_indices = indices[num_training:]

    train_dataset = CSI_Dataset(mat_dataset_path, indices=train_indices)
    val_dataset = CSI_Dataset(mat_dataset_path, indices=val_indices)

    # num_workers 设置为 0 避免 Windows 下的多进程读取 HDF5 报错
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


# ==========================================
# 3. 核心训练逻辑
# ==========================================
def train_model(model_name, dataset_train_path, dict_path, num_layers=15, epochs=100, lr=1e-4,
                batch_size=64, num_training_samples=None, model_path=None,
                load_latest_state=False, device='cuda', alpha_cfo=0.1):
    loader_train, loader_val = training_setup(dataset_train_path, num_training_samples, batch_size)
    array_type = "4t4r"

    # 初始化网络 (内部已包含字典加载和 2D-FFT 特征值计算)
    network = initialize_model(model_name, dict_path, num_layers, device)

    # 二次确保整个网络参数处于 float32 模式
    network = network.to(device).float()

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    if load_latest_state or model_path:
        try:
            load_state(network, optimizer, array_type=array_type,
                       load_latest_state=load_latest_state, model_path=model_path)
        except Exception as e:
            print(f"未能加载预训练权重 ({e})，将从头开始训练。")

    print(f"\n开始训练 | 架构: {model_name} (One-shot) | CFO权重: {alpha_cfo}")

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        network.train()
        epoch_spec_loss = 0
        epoch_cfo_loss = 0

        for yf, x_gt, labels in loader_train:
            # 将数据送入 GPU
            yf = yf.to(device)
            x_gt = x_gt.to(device)
            cfo_gt = labels['cfo'].to(device)

            # --- 前向传播 ---
            # 返回: 估计的纯净 CSI (x_est) 和 预测的 CFO (cfo_out)
            res = network(yf, return_cfo=True)

            # 兼容处理: 检查返回值是元组还是单个张量
            if isinstance(res, tuple):
                x_est, cfo_out = res[0], res[1]
            else:
                x_est, cfo_out = res, None

            # --- A. 谱图重构损失 (NMSE) ---
            diff = x_est - x_gt
            num = torch.sum(torch.abs(diff) ** 2, dim=(1, 2))
            den = torch.sum(torch.abs(x_gt) ** 2, dim=(1, 2)) + 1e-12  # +1e-12 防止除零
            loss_spec = (num / den).mean()

            # --- B. CFO 物理损失 (MSE) ---
            loss_cfo = torch.tensor(0.0, device=device)
            if cfo_out is not None:
                # 兼容多层迭代输出列表 或 单一输出张量
                if isinstance(cfo_out, list):
                    for df in cfo_out:
                        loss_cfo += F.mse_loss(df.squeeze(), cfo_gt)
                    loss_cfo = loss_cfo / len(cfo_out)
                else:
                    loss_cfo = F.mse_loss(cfo_out.squeeze(), cfo_gt)

                total_loss = loss_spec + alpha_cfo * loss_cfo
            else:
                total_loss = loss_spec

            # --- 反向传播 ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_spec_loss += loss_spec.item()
            if cfo_out is not None:
                epoch_cfo_loss += loss_cfo.item()

        # --- 验证环节 ---
        network.eval()
        val_nmse_total = 0
        with torch.no_grad():
            for yfv, x_gtv, _ in loader_val:
                yfv = yfv.to(device)
                x_gtv = x_gtv.to(device)

                res_v = network(yfv, return_cfo=False)
                x_estv = res_v[0] if isinstance(res_v, tuple) else res_v

                num_v = torch.sum(torch.abs(x_estv - x_gtv) ** 2, dim=(1, 2))
                den_v = torch.sum(torch.abs(x_gtv) ** 2, dim=(1, 2)) + 1e-12
                val_nmse_total += (num_v / den_v).mean().item()

        # 计算并记录 Epoch 平均损失
        avg_train_nmse = epoch_spec_loss / len(loader_train)
        avg_val_nmse = val_nmse_total / len(loader_val)
        avg_cfo_err = epoch_cfo_loss / len(loader_train)

        train_loss_history.append(avg_train_nmse)
        val_loss_history.append(avg_val_nmse)

        # 打印进度 (转为 dB 显示更直观)
        print(f"Epoch [{epoch + 1:03d}/{epochs}] "
              f"NMSE: {10 * torch.log10(torch.tensor(avg_train_nmse)):.2f}dB | "
              f"Val: {10 * torch.log10(torch.tensor(avg_val_nmse)):.2f}dB | "
              f"CFO-MSE: {avg_cfo_err:.6f}")

    # 训练结束，保存模型
    save_state(network, optimizer, array_type, train_loss_history, val_loss_history)
    return network


# ==========================================
# 4. 模型状态保存
# ==========================================
def save_state(network, optimizer, array_type, train_loss, val_loss):
    model_name, model_tag = model_id(network)
    num_layers = getattr(network, 'num_layers', 15)

    state_dir = os.path.join(STATES_PATH, array_type, model_name, f"{num_layers}l")
    loss_dir = os.path.join(LOSSES_PATH, array_type, model_name, f"{num_layers}l")
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    save_path = os.path.join(state_dir, f"{model_tag}.pt")
    torch.save({
        'model_state': network.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'model_tag': model_tag
    }, save_path)

    torch.save({
        'training_loss': train_loss,
        'validation_loss': val_loss
    }, os.path.join(loss_dir, f"{model_tag}_loss.pt"))

    print(f"\n训练结束。模型已保存至: {save_path}")