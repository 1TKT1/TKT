import os
import sys
import torch
import h5py
from models.cadmmnet import CAdmmNet

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATES_PATH = f"{BASE_PATH}/models/states"

def initialize_model(model, dictionary_path, num_layers, device="cuda"):
    print(f"正在加载物理字典: {dictionary_path}")
    with h5py.File(dictionary_path, 'r') as f:
        A_theta = torch.complex(
            torch.from_numpy(f['A_theta']['real'][:]).to(torch.float32),
            torch.from_numpy(f['A_theta']['imag'][:]).to(torch.float32)
        ).T.to(device)
        A_tau = torch.complex(
            torch.from_numpy(f['A_tau']['real'][:]).to(torch.float32),
            torch.from_numpy(f['A_tau']['imag'][:]).to(torch.float32)
        ).T.to(device)

    M, K = A_theta.shape[0], A_tau.shape[0]


    A_theta = A_theta / (M ** 0.5)
    A_tau = A_tau / (K ** 0.5)


    B_theta = torch.matmul(A_theta.T.conj(), A_theta)
    B_tau = torch.matmul(A_tau.T.conj(), A_tau)

    mid_theta = B_theta[:, B_theta.shape[0] // 2]
    mid_tau = B_tau[:, B_tau.shape[0] // 2]
    base_theta = torch.fft.ifftshift(mid_theta)
    base_tau = torch.fft.ifftshift(mid_tau)

    b_2d_0 = torch.outer(base_theta, base_tau)
    # FFT 之后再取 real 是正确的，因为共轭对称序列的傅里叶变换是纯实数
    V_2d = torch.fft.fft2(b_2d_0).real.to(torch.float32)

    beta = 0.2

    if model == "CADMM-Net":
        network = CAdmmNet(V_2d, A_theta, A_tau, num_layers, beta, 1.0, device)
    else:
        raise ValueError("仅支持 2D CADMM-Net。")
    return network.to(device).float()

def load_state(network, optimizer=None, array_type="2d_ofdm", load_latest_state=True, model_path=None, return_tag=False):
    if load_latest_state:
        model_name = network.__class__.__name__.replace("-", "").lower()
        num_layers = getattr(network, 'num_layers', 15)
        state_path = f"{STATES_PATH}/{array_type}/{model_name}/{num_layers}l"
        model_id_prefix = f"{model_name}_{num_layers}l"
        os.makedirs(state_path, exist_ok=True)
        matching_states = [f.name for f in os.scandir(state_path) if f.is_file() and f.name.startswith(model_id_prefix)]
        if not matching_states:
            raise FileNotFoundError("未找到模型权重。")
        latest = max(matching_states, key=lambda f: os.path.getmtime(os.path.join(state_path, f)))
        model_path = f"{state_path}/{latest}"

    device = next(network.parameters()).device
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    network.load_state_dict(checkpoint["model_state"], strict=False)
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if return_tag:
        return checkpoint.get("model_tag", os.path.basename(model_path).replace('.pt', ''))