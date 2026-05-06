import os
import sys
import torch

# 假设你的 CAdmmNet 定义在同级的 models 文件夹下
from models import CAdmmNet

utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(utils_dir)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATES_PATH = f"{BASE_PATH}/models/states"


def initialize_model(model, dictionary_path, num_layers, device="cuda"):
    """
    初始化 CADMM-Net 模型，并利用物理字典预计算 BCCB 矩阵的频域特征值。
    """
    if not os.path.exists(dictionary_path):
        raise FileNotFoundError(f"字典文件未找到: {dictionary_path}")

    print(f"正在加载物理字典: {dictionary_path}")
    # 加入 map_location 确保加载到的设备与当前设定一致
    dict_data = torch.load(dictionary_path, map_location=device, weights_only=True)
    A_theta = dict_data['A_theta'].to(device)
    A_tau = dict_data['A_tau'].to(device)

    # 1. 计算一维 Gram 矩阵 (自相关矩阵)
    B_theta = torch.matmul(A_theta.T.conj(), A_theta)
    B_tau = torch.matmul(A_tau.T.conj(), A_tau)

    # 2. 提取第一列，外积生成 BCCB 矩阵的基础块 (表示 2D 联合阵列流形的特性)
    b_2d_0 = torch.outer(B_theta[:, 0], B_tau[:, 0])

    # 3. 2D-FFT 提取二维频域特征值，并取实部
    # 【关键修复】: 必须使用 torch.float32，以匹配 CNN 和输入数据的精度，防止 double/float 报错
    V_2d = torch.fft.fft2(b_2d_0).real.to(torch.float32)

    # 算法超参数初始化 (可根据论文设定调整)
    rho = 1.0
    beta = 1e-1

    if model == "CADMM-Net":
        network = CAdmmNet(V_2d, num_layers, beta, rho, device)
    else:
        raise ValueError("This version is specialized for 2D CADMM-Net.")

    # 再次确保整个网络的所有参数都严格处于 float32 精度
    return network.to(device).float()


def load_state(network, optimizer=None, array_type="4t4r", load_latest_state=True, model_path=None, return_tag=False):
    """
    模型状态加载器，支持自动寻找最新断点续训。
    """
    if load_latest_state:
        model_name = network.__class__.__name__.replace("-", "").lower()
        num_layers = getattr(network, 'num_layers', 15)
        state_path = f"{STATES_PATH}/{array_type}/{model_name}/{num_layers}l"
        model_id_prefix = f"{model_name}_{num_layers}l"

        if not os.path.exists(state_path):
            raise FileNotFoundError(f"找不到模型状态目录: {state_path}")

        matching_states = [state.name for state in os.scandir(state_path)
                           if state.is_file() and state.name.startswith(model_id_prefix)]

        if not matching_states:
            raise FileNotFoundError(f"目录 {state_path} 下没有找到匹配的模型权重文件。")

        # 按照修改时间排序，拿到最新训练出的权重
        latest_model_state = max(matching_states, key=lambda f: os.path.getmtime(os.path.join(state_path, f)))
        model_path = f"{state_path}/{latest_model_state}"
        print(f"自动加载最新模型权重: '{model_path}' ...")
    else:
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"指定的模型路径无效: {model_path}")
        print(f"加载指定模型权重: '{model_path}' ...")

    # 获取网络所在的设备，用于安全加载
    device = next(network.parameters()).device

    # 读取检查点
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # 恢复网络权重
    network.load_state_dict(checkpoint["model_state"])

    # 恢复优化器状态 (仅在训练/断点续训时需要)
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if return_tag:
        return checkpoint.get("model_tag", "unknown_tag")