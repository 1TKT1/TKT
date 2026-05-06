import argparse
import json
import torch

# 🌟 修复：只导入我们需要的 generate_dictionary，去掉了被废弃的函数
from datasets.dataset_generator import generate_dictionary
from utils.training_utils import train_model
from utils.utils import set_random_seeds
from utils.metric_utils import evaluate_model

SEED = 1234

def main():
    """
    主脚本：管理二维数据集生成、模型训练和测试。
    注意：当前架构下，大规模训练数据已由 MATLAB 生成 (.mat)，
    Python 端主要负责生成字典 (.pt) 以及执行训练和评估。
    """
    parser = argparse.ArgumentParser(
        description="Main script to manage 2D dataset generation, training, and testing of models.")
    parser.add_argument('--config', type=str, help='Path to JSON config file.')
    subparsers = parser.add_subparsers(dest='command', help="Choose between dataset generation, training, and testing.")

    # --- 命令 1: 仅生成字典 (适配 MATLAB 数据的关键步骤) ---
    parser_array = subparsers.add_parser("create-array", help="Generate dictionary only.")
    parser_array.add_argument('--array_type', type=str, default='4T4R')
    parser_array.add_argument('--num_elements', type=int, default=4)     # 4 天线
    parser_array.add_argument('--num_subcarriers', type=int, default=432) # 432 子载波
    parser_array.add_argument('--N_theta', type=int, default=128)
    parser_array.add_argument('--N_tau', type=int, default=128)

    # --- 命令 2: 训练模型 ---
    parser_train = subparsers.add_parser("train-model", help="Train CADMM-Net.")
    parser_train.add_argument('--model', type=str, default='CADMM-Net')
    parser_train.add_argument('--num_layers', type=int, default=15)
    parser_train.add_argument('--dataset_train_path', type=str, required=True, help="Path to the MATLAB .mat dataset")
    parser_train.add_argument('--dict_path', type=str, required=True, help="Path to the Python generated .pt dictionary")
    parser_train.add_argument('--epochs', type=int, default=100)
    parser_train.add_argument('--lr', type=float, default=1e-4)
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--num_training_samples', type=int, default=None)
    parser_train.add_argument('--model_path', type=str, default=None)
    parser_train.add_argument('--load_latest_state', action='store_true')
    parser_train.add_argument('--device', type=str, default='cuda')
    parser_train.add_argument('--alpha_cfo', type=float, default=0.0, help="Weight for CFO loss. Default 0 for clean data.")

    # --- 命令 3: 评估模型 ---
    parser_metrics = subparsers.add_parser("evaluate-model", help="Evaluate performance.")
    parser_metrics.add_argument('--model', type=str, default='CADMM-Net')
    parser_metrics.add_argument('--num_layers', type=int, default=15)
    parser_metrics.add_argument('--dataset_test_path', type=str, required=True)
    parser_metrics.add_argument('--dict_path', type=str, default=None, help="Path to dictionary if needed by evaluator")
    parser_metrics.add_argument('--model_path', type=str, default=None)
    parser_metrics.add_argument('--load_latest_state', type=bool, default=True, help="Automatically loads the latest saved state.")
    parser_metrics.add_argument('--metric', type=str, default='detection_rate')
    parser_metrics.add_argument('--bin_threshold', type=int, default=2)
    parser_metrics.add_argument('--amp_threshold', type=float, default=0.4)
    parser_metrics.add_argument('--return_degs', type=bool, default=True)
    parser_metrics.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    set_random_seeds(SEED)

    # 处理 JSON 配置文件
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            command_args = config.get(args.command, {})
            for key, value in command_args.items():
                setattr(args, key, value)

    # 逻辑分发
    if args.command == 'create-array':
        generate_dictionary(
            array_type=args.array_type,
            num_elements=args.num_elements,
            num_subcarriers=args.num_subcarriers,
            N_theta=args.N_theta,
            N_tau=args.N_tau
        )

    elif args.command == 'train-model':
        train_model(
            model_name=args.model,
            dataset_train_path=args.dataset_train_path,
            dict_path=args.dict_path,
            num_layers=args.num_layers,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_training_samples=args.num_training_samples,
            model_path=args.model_path,
            load_latest_state=args.load_latest_state,
            device=args.device,
            alpha_cfo=args.alpha_cfo
        )

    elif args.command == 'evaluate-model':
        evaluate_model(
            model=args.model,
            dataset_test_path=args.dataset_test_path,
            num_layers=args.num_layers,
            model_path=args.model_path,
            load_latest_state=args.load_latest_state,
            metric=args.metric,
            bin_threshold=args.bin_threshold,
            amp_threshold=args.amp_threshold,
            return_degs=args.return_degs,
            device=args.device
        )

if __name__ == '__main__':
    main()