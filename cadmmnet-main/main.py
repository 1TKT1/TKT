import argparse
import json
import sys  # 🚀 修复点 1：引入 Python 标准系统模块
import torch
from utils.training_utils import train_model
from utils.metric_utils import evaluate_model
from utils.utils import set_random_seeds

SEED = 1234


def main():
    parser = argparse.ArgumentParser(description="2D CADMM-Net Trainer & Evaluator")
    parser.add_argument('--config', type=str, help='Path to JSON config file.')
    subparsers = parser.add_subparsers(dest='command')

    # ==========================================
    # 1. 注册 train-model 命令
    # ==========================================
    parser_train = subparsers.add_parser("train-model", help="Train CADMM-Net.")
    parser_train.add_argument('--model', type=str, default='CADMM-Net')
    parser_train.add_argument('--num_layers', type=int, default=15)
    parser_train.add_argument('--dataset_train_path', type=str)
    parser_train.add_argument('--epochs', type=int, default=60)
    parser_train.add_argument('--lr', type=float, default=1e-4)
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--num_training_samples', type=int, default=1800)
    parser_train.add_argument('--device', type=str, default='cuda')

    # ==========================================
    # 2. 注册 evaluate-model 命令
    # ==========================================
    parser_eval = subparsers.add_parser("evaluate-model", help="Evaluate CADMM-Net RMSE.")
    parser_eval.add_argument('--model', type=str, default='CADMM-Net')
    parser_eval.add_argument('--num_layers', type=int, default=15)
    parser_eval.add_argument('--dataset_test_path', type=str)
    parser_eval.add_argument('--model_path', type=str, default=None)
    parser_eval.add_argument('--load_latest_state', type=bool, default=True)
    parser_eval.add_argument('--metric', type=str, default='rmse')
    parser_eval.add_argument('--bin_threshold', type=int, default=2)
    parser_eval.add_argument('--amp_threshold', type=float, default=0.4)
    parser_eval.add_argument('--device', type=str, default='cuda')

    # 显式捕获原始命令行输入，用于后续最高优先级覆盖
    raw_args = parser.parse_known_args()[0]
    args = parser.parse_args()
    set_random_seeds(SEED)

    # 从 config.json 读取基础参数
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            command_args = config.get(args.command, {})
            for key, value in command_args.items():
                setattr(args, key, value)



        for action in parser_eval._actions if args.command == 'evaluate-model' else parser_train._actions:
            arg_name = action.dest
            if hasattr(raw_args, arg_name) and getattr(raw_args, arg_name) is not None:
                # 排除 subparser 默认产生的默认值干扰
                if arg_name != 'command' and f'--{arg_name}' in ''.join(sys.argv):
                    setattr(args, arg_name, getattr(raw_args, arg_name))

    # ==========================================
    # 3. 根据命令执行对应的函数
    # ==========================================
    if args.command == 'train-model':
        model_name_to_use = getattr(args, 'model_name', args.model)
        train_model(
            model=model_name_to_use,
            dataset_train_path=args.dataset_train_path,
            num_layers=args.num_layers,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_training_samples=args.num_training_samples,
            device=args.device
        )

    elif args.command == 'evaluate-model':
        if args.model_path is not None:
            args.load_latest_state = False

        model_name_to_use = getattr(args, 'model_name', args.model)
        evaluate_model(
            model=model_name_to_use,
            dataset_test_path=args.dataset_test_path,
            num_layers=args.num_layers,
            model_path=args.model_path,
            load_latest_state=args.load_latest_state,
            metric=args.metric,
            bin_threshold=args.bin_threshold,
            amp_threshold=args.amp_threshold,
            device=args.device
        )


if __name__ == '__main__':
    main()