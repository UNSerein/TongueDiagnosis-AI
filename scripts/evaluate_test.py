# evaluate_test.py
# 用于对指定 YOLOv8 实验在 test 集上进行评估，并保存结果到该实验目录
# 自动从 args.yaml 读取训练时的 imgsz，确保评估尺寸一致

import argparse
from pathlib import Path
import yaml
import json
from datetime import datetime
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on test set and save results")
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name under runs/detect/ (e.g. exp_800_disk_ep30)')
    parser.add_argument('--exp-dir', type=str, default=None,
                        help='Full path to the experiment folder (overrides --exp-name)')
    parser.add_argument('--data', type=str, default='data/dataset.yaml',
                        help='Path to dataset.yaml')
    parser.add_argument('--output-file', type=str, default='test_results.txt',
                        help='Output file name inside the exp directory')
    parser.add_argument('--save-json', action='store_true',
                        help='Also save detailed results as JSON')
    return parser.parse_args()


def main():
    args = parse_args()

    # 确定实验目录
    if args.exp_dir:
        exp_path = Path(args.exp_dir).resolve()
    elif args.exp_name:
        exp_path = Path("runs/detect") / args.exp_name
    else:
        print("Error: Must provide either --exp-name or --exp-dir")
        return

    if not exp_path.exists():
        print(f"Error: Experiment directory not found: {exp_path}")
        return

    # 寻找 args.yaml 以读取训练时的 imgsz
    args_yaml_path = None
    for p in [exp_path / "args.yaml", exp_path / "train" / "args.yaml"]:
        if p.exists():
            args_yaml_path = p
            break

    if args_yaml_path:
        try:
            with open(args_yaml_path, 'r', encoding='utf-8') as f:
                train_args = yaml.safe_load(f)
            eval_imgsz = train_args.get('imgsz', 640)
            print(f"Detected training imgsz from args.yaml: {eval_imgsz}")
        except Exception as e:
            print(f"Warning: Failed to read args.yaml: {e}")
            eval_imgsz = 640
    else:
        print("Warning: args.yaml not found in experiment directory, using default imgsz=640")
        eval_imgsz = 640

    # 寻找模型权重（优先 best.pt，其次 last.pt）
    weights_candidates = [
        exp_path / "train" / "weights" / "best.pt",
        exp_path / "weights" / "best.pt",
        exp_path / "train" / "weights" / "last.pt",
        exp_path / "weights" / "last.pt",
    ]

    model_path = None
    for candidate in weights_candidates:
        if candidate.exists():
            model_path = candidate
            print(f"Found model: {model_path}")
            break

    if not model_path:
        print("Error: No best.pt or last.pt found in the experiment directory")
        print("Checked paths:")
        for p in weights_candidates:
            print(f"  - {p}")
        return

    # 加载模型
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))

    # 在 test 集上进行验证
    print("Starting evaluation on test set...")
    results = model.val(
        data=args.data,
        split="test",
        imgsz=eval_imgsz,           # 使用从 args.yaml 读取的尺寸
        batch=16,
        workers=4,
        device=0,
        plots=False,
        save=False,
        save_txt=False,
        save_conf=False,
        verbose=True
    )

    # 提取主要指标
    metrics = results.results_dict
    eval_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 用当前时间代替不存在的 dates

    summary = {
        "model": str(model_path),
        "mAP50": float(metrics.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(metrics.get("metrics/precision(B)", 0.0)),
        "recall": float(metrics.get("metrics/recall(B)", 0.0)),
        "fitness": float(metrics.get("fitness", 0.0)),
        "eval_date": eval_date,
        "eval_imgsz": eval_imgsz,
    }

    # 打印到终端
    print("\n" + "="*60)
    print("Test Set Evaluation Results")
    print("="*60)
    print(f"Model          : {summary['model']}")
    print(f"Eval imgsz     : {summary['eval_imgsz']}")
    print(f"mAP@0.5        : {summary['mAP50']:.4f}")
    print(f"mAP@0.5:0.95   : {summary['mAP50-95']:.4f}")
    print(f"Precision      : {summary['precision']:.4f}")
    print(f"Recall         : {summary['recall']:.4f}")
    print(f"Fitness        : {summary['fitness']:.4f}")
    print(f"Evaluation date: {summary['eval_date']}")
    print("="*60)

    # 保存到 txt 文件
    output_path = exp_path / args.output_file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("YOLOv8 Test Set Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Model path     : {summary['model']}\n")
        f.write(f"Eval imgsz     : {summary['eval_imgsz']}\n")
        f.write(f"Evaluation date: {summary['eval_date']}\n")
        f.write(f"mAP@0.5        : {summary['mAP50']:.4f}\n")
        f.write(f"mAP@0.5:0.95   : {summary['mAP50-95']:.4f}\n")
        f.write(f"Precision      : {summary['precision']:.4f}\n")
        f.write(f"Recall         : {summary['recall']:.4f}\n")
        f.write(f"Fitness        : {summary['fitness']:.4f}\n")
        f.write("\nCommand used:\n")
        f.write(f"  python evaluate_test.py --exp-name {args.exp_name or exp_path.name}\n")

    print(f"Results saved to: {output_path}")

    # 可选：保存详细 JSON
    if args.save_json:
        json_path = exp_path / "test_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "full_metrics": metrics
            }, f, indent=4, ensure_ascii=False)
        print(f"Detailed JSON saved to: {json_path}")


if __name__ == "__main__":
    main()