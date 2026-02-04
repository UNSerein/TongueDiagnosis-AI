import os
import shutil
import argparse
from datetime import datetime
from ultralytics import YOLO

# =========================
# 基本配置（可通过命令行修改）
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script for Tongue Diagnosis")
    parser.add_argument('--model', type=str, default='models/yolov8n.pt', help='Path to model weights (e.g., yolov8n.pt)')
    parser.add_argument('--data', type=str, default='data/dataset.yaml', help='Path to dataset.yaml')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size (adjust based on memory)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='cpu', help='Device: "cpu" or "0" for GPU')
    parser.add_argument('--exp_dir', type=str, default='experiments/exp1_yolov8n_cpu_ep5', help='Experiment directory')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers')
    return parser.parse_args()

args = parse_args()

# =========================
# 训练函数
# =========================
def train_yolo():
    # 加载模型
    model = YOLO(args.model)

    # 创建实验目录
    results_dir = os.path.join(args.exp_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 训练（指定project和name来控制保存路径）
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=results_dir,  # 保存到 experiments/xxx/results/runs
        name='train',         # 子文件夹名
        exist_ok=True         # 覆盖现有
    )

    # 训练完成后，复制关键文件到results_dir（YOLO默认保存到 runs/detect/train）
    run_dir = os.path.join(results_dir, 'runs', 'detect', 'train')
    if os.path.exists(run_dir):
        # 复制指标、权重等
        shutil.copy(os.path.join(run_dir, 'results.csv'), results_dir)
        shutil.copy(os.path.join(run_dir, 'weights/best.pt'), results_dir)
        shutil.copy(os.path.join(run_dir, 'weights/last.pt'), results_dir)
        shutil.copy(os.path.join(run_dir, 'confusion_matrix.png'), results_dir)
        shutil.copy(os.path.join(run_dir, 'results.png'), results_dir)

    # 生成README总结
    with open(os.path.join(args.exp_dir, 'README.md'), 'a') as f:
        f.write(f"\n## Training Summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"- Model: {args.model}\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Device: {args.device}\n")
        f.write(f"- Batch Size: {args.batch}\n")
        f.write(f"- Results saved to: {results_dir}\n")

    print(f"Training completed. Results saved to: {results_dir}")

if __name__ == "__main__":
    train_yolo()