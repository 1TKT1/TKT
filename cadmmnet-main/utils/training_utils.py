import os
import h5py
import torch
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from utils.initialization_utils import initialize_model, load_state
from utils.utils import model_id

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATES_PATH = os.path.join(BASE_PATH, "models", "states")


def generate_gaussian_kernel_2d(kernel_size=7, sigma=0.5, device='cuda'):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g).unsqueeze(0).unsqueeze(0)



class CSI_Dataset(Dataset):
    def __init__(self, mat_path, indices=None):
        self.mat_path = mat_path
        self.indices = indices
        self.file = None  # 初始化为 None，绝对不要在初始化时打开文件
        self._length = None  # 用于缓存数据集长度

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)

        # 仅在第一次查询长度时短暂打开一次
        if self._length is None:
            with h5py.File(self.mat_path, 'r') as f:
                self._length = f['all_H_cfr']['real'].shape[0]
        return self._length

    def __getitem__(self, idx):

        # 这样训练集和验证集各自维护稳定的单一长连接，避免高频开关带来的 OS 锁死
        if self.file is None:
            self.file = h5py.File(self.mat_path, 'r')

        actual_idx = self.indices[idx] if self.indices is not None else idx

        obs_real = self.file['all_H_cfr']['real'][actual_idx]
        obs_imag = self.file['all_H_cfr']['imag'][actual_idx]
        label = self.file['all_X_label'][actual_idx]

        yf = torch.complex(
            torch.from_numpy(obs_real).to(torch.float32),
            torch.from_numpy(obs_imag).to(torch.float32)
        ).T
        x_gt = torch.from_numpy(label).to(torch.float32).T

        return yf, x_gt

    def __del__(self):
        # 当 Dataset 对象被垃圾回收销毁时，安全关闭底层 HDF5 文件句柄
        if hasattr(self, 'file') and self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass


def training_setup(mat_dataset_path, num_training_samples=None, batch_size=32):
    with h5py.File(mat_dataset_path, 'r') as f:
        total_samples = f['all_H_cfr']['real'].shape[0]
    num_training = int(num_training_samples) if num_training_samples else int(total_samples * 0.9)
    indices = list(range(total_samples))
    train_loader = DataLoader(CSI_Dataset(mat_dataset_path, indices[:num_training]), batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(CSI_Dataset(mat_dataset_path, indices[num_training:]), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(model, dataset_train_path, num_layers=15, epochs=150, lr=1e-4, batch_size=32, num_training_samples=1800,
                model_path=None, load_latest_state=False, device='cuda'):
    loader_train, loader_val = training_setup(dataset_train_path, num_training_samples, batch_size)
    network = initialize_model(model, dataset_train_path, num_layers, device).to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    KERNEL_SIZE = 11
    KERNEL_SCALE = 1.5
    kernel_2d = generate_gaussian_kernel_2d(KERNEL_SIZE, KERNEL_SCALE, device)
    padding = KERNEL_SIZE // 2

    # 创建保存目录，准备“边跑边存”
    model_name, model_tag = model_id(network)
    state_dir = os.path.join(STATES_PATH, "2d_ofdm", model_name, f"{num_layers}l")
    os.makedirs(state_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\n 启动训练 | 架构: {model} | 目标: 2D 高斯平滑 NMSE 损失")

    for epoch in range(epochs):
        network.train()
        epoch_loss = 0
        for yf, x_gt in loader_train:
            yf, x_gt = yf.to(device), x_gt.to(device)
            x_est = network(yf)

            x_est_un = x_est.unsqueeze(1)
            x_gt_un = x_gt.unsqueeze(1)
            x_est_pad = F.pad(x_est_un, (padding, padding, padding, padding), mode='reflect')
            x_gt_pad = F.pad(x_gt_un, (padding, padding, padding, padding), mode='reflect')

            x_est_blurred = F.conv2d(x_est_pad, kernel_2d).squeeze(1)
            x_gt_blurred = F.conv2d(x_gt_pad, kernel_2d).squeeze(1)

            se = torch.sum((x_est_blurred - x_gt_blurred) ** 2, dim=(1, 2))
            pw = torch.sum(x_gt_blurred ** 2, dim=(1, 2))
            loss = torch.mean(se / (pw + 1e-12))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        network.eval()
        val_loss = 0
        with torch.no_grad():
            for yfv, x_gtv in loader_val:
                yfv, x_gtv = yfv.to(device), x_gtv.to(device)
                x_estv = network(yfv)

                x_estv_pad = F.pad(x_estv.unsqueeze(1), (padding, padding, padding, padding), mode='reflect')
                x_gtv_pad = F.pad(x_gtv.unsqueeze(1), (padding, padding, padding, padding), mode='reflect')

                v_est_blurred = F.conv2d(x_estv_pad, kernel_2d).squeeze(1)
                v_gt_blurred = F.conv2d(x_gtv_pad, kernel_2d).squeeze(1)

                v_se = torch.sum((v_est_blurred - v_gt_blurred) ** 2, dim=(1, 2))
                v_pw = torch.sum(v_gt_blurred ** 2, dim=(1, 2))
                val_loss += torch.mean(v_se / (v_pw + 1e-12)).item()

        avg_train = epoch_loss / len(loader_train)
        avg_val = val_loss / len(loader_val)
        current_lr = optimizer.param_groups[0]['lr']

        # 一旦创下历史新低，立刻保存模型权重
        is_best = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'model_state': network.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'model_tag': model_tag
            }, os.path.join(state_dir, f"{model_tag}.pt"))
            is_best = "  [New Best Saved]"

        print(
            f"Epoch [{epoch + 1:03d}/{epochs}] "
            f"LR: {current_lr:.2e} | "
            f"Train: {10 * torch.log10(torch.tensor(avg_train)):.2f} dB | "
            f"Val: {10 * torch.log10(torch.tensor(avg_val)):.2f} dB{is_best}"
        )

    print(f"\n 训练完毕！最优权重已安全锁定于: {state_dir}")