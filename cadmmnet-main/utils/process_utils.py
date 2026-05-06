import os
import glob
import torch

# 获取基础路径
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_PATH = f"{BASE_PATH}/outputs/metrics"
LOSSES_PATH = f"{BASE_PATH}/outputs/losses"
SPECTRUMS_PATH = f"{BASE_PATH}/outputs/spectrums"


def _load_pt_files(folder_path, load_all, snr_values_ref):
    """
    内部辅助函数：加载指定文件夹下的 .pt 文件
    :param folder_path: 目标文件夹路径
    :param load_all: 是否加载全部历史文件 (False 则只加载最新)
    :param snr_values_ref: 用于保存/更新全局 SNR 数组的列表引用
    :return: 解析后的字典 {tag: data} 或 单一数据
    """
    pt_files = glob.glob(os.path.join(folder_path, "*.pt"))
    if not pt_files:
        return None

    # 选择最新文件或全部文件
    selected_files = [max(pt_files, key=os.path.getmtime)] if not load_all else pt_files

    extracted_data = {}
    for pt_file in selected_files:
        try:
            # 强制使用 weights_only=True 以提升安全性和速度
            data = torch.load(pt_file, weights_only=True)

            # 提取共有参数：信噪比数组
            if not snr_values_ref:
                snr_values_ref.extend(data.get("snr_values", []))

            # 提取文件名作为 Tag
            tag = os.path.splitext(os.path.basename(pt_file))[0]

            # 针对不同数据类型提取对应的内容
            if "training_loss" in data:
                m_val = {
                    "training_loss": data.get("training_loss"),
                    "validation_loss": data.get("validation_loss")
                }
            elif "spectrums" in data:
                m_val = {
                    "spectrums": data.get("spectrums"),
                    "ground_truth": data.get("ground_truth")
                }
            else:
                m_val = data.get("average")

            extracted_data[tag] = m_val

        except Exception as e:
            print(f"Error loading {pt_file}: {e}")

    # 如果只加载一个，直接返回值而不是字典
    if not load_all and extracted_data:
        return list(extracted_data.values())[0]

    return extracted_data


def load_output(output_type, load_all=False):
    """
    专门适配 2D Joint TDE-DOA 任务的输出加载器。
    支持的 output_type：'metrics', 'losses', 'spectrums'
    """
    paths = {
        "metrics": METRICS_PATH,
        "losses": LOSSES_PATH,
        "spectrums": SPECTRUMS_PATH
    }

    base_dir = paths.get(output_type)
    if base_dir is None:
        raise ValueError(f"Invalid output type: {output_type}")

    results = {}
    snr_values_ref = []  # 使用列表引用来跨函数保存 SNR

    if not os.path.exists(base_dir):
        print(f"⚠️ Warning: Path {base_dir} does not exist.")
        return results

    # 1. 遍历 Array Type (例如: '2d_ula', '4t4r')
    for array_type in os.listdir(base_dir):
        array_path = os.path.join(base_dir, array_type)
        if not os.path.isdir(array_path):
            continue
        results[array_type] = {}

        if output_type == "metrics":
            # 2. 遍历指标类型 (例如: 'nmse', 'rmse_doa')
            for metric in os.listdir(array_path):
                metric_path = os.path.join(array_path, metric)
                if not os.path.isdir(metric_path):
                    continue
                results[array_type][metric] = {}

                # 3. 遍历模型 (例如: 'CADMM-Net')
                for model_name in os.listdir(metric_path):
                    model_path = os.path.join(metric_path, model_name)
                    if not os.path.isdir(model_path):
                        continue
                    results[array_type][metric][model_name] = {}

                    # 4. 遍历层数 (例如: '15l')
                    for layer_folder in os.listdir(model_path):
                        layer_path = os.path.join(model_path, layer_folder)
                        if not os.path.isdir(layer_path):
                            continue
                        num_layers = layer_folder.rstrip("l")

                        # 物理指标通常需要 Thresh 过滤背景
                        has_thresh = metric in ["rmse_doa", "rmse_tde", "detection_rate", "rmse"]

                        if has_thresh:
                            results[array_type][metric][model_name][num_layers] = {}
                            # 5. 遍历阈值文件夹 (例如: 'thresh_0.4')
                            for thresh_folder in os.listdir(layer_path):
                                thresh_path = os.path.join(layer_path, thresh_folder)
                                if not os.path.isdir(thresh_path):
                                    continue

                                data = _load_pt_files(thresh_path, load_all, snr_values_ref)
                                if data is not None:
                                    results[array_type][metric][model_name][num_layers][thresh_folder] = data
                        else:
                            # NMSE 无需阈值层级，直接加载
                            data = _load_pt_files(layer_path, load_all, snr_values_ref)
                            if data is not None:
                                results[array_type][metric][model_name][num_layers] = data

        else:
            # 针对 'losses' 和 'spectrums'
            for model_name in os.listdir(array_path):
                model_path = os.path.join(array_path, model_name)
                if not os.path.isdir(model_path):
                    continue
                results[array_type][model_name] = {}

                for layer_folder in os.listdir(model_path):
                    layer_path = os.path.join(model_path, layer_folder)
                    if not os.path.isdir(layer_path):
                        continue
                    num_layers = layer_folder.rstrip("l")

                    data = _load_pt_files(layer_path, load_all, snr_values_ref)
                    if data is not None:
                        results[array_type][model_name][num_layers] = data

    # 提取全局 SNR (如果存在)
    if snr_values_ref:
        results["snr"] = snr_values_ref

    return results