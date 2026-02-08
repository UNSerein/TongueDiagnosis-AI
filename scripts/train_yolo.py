# 该脚本使用Ultralytics库训练用于舌头诊断的YOLOv8模型。
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script for Tongue Diagnosis")
    parser.add_argument('--model', type=str, default='models/yolov8n.pt',
                        help='Path to pretrained model weights (e.g. yolov8n.pt, yolov8s.pt)')
    parser.add_argument('--data', type=str, default='data/dataset.yaml',
                        help='Path to dataset yaml file')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=float, default=-1.0,
                        help='Batch size: int (e.g. 16), -1 for AutoBatch, float like 0.8 for 80%% memory')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (pixels)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device: "cpu" or GPU index like "0", "0,1"')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loader workers')
    parser.add_argument('--cache', action='store_true',
                        help='Cache images to RAM (faster) or disk')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom name for this run (saved in runs/detect/<name>)')
    parser.add_argument('--cache-type', type=str, default='ram', choices=['ram', 'disk'],
                        help='Cache type when --cache is enabled: ram or disk')
    
    return parser.parse_args()


def main():
    args = parse_args()

    print("Starting YOLOv8 training with following parameters:")
    print(f"  Model:     {args.model}")
    print(f"  Data:      {args.data}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch} (auto if -1)")
    print(f"  Image size:{args.imgsz}")
    print(f"  Device:    {args.device}")
    print(f"  Workers:   {args.workers}")
    print(f"  Cache:     {args.cache} ({args.cache_type if args.cache else 'disabled'})")
    if args.name:
        print(f"  Run name:  {args.name}")

    # 加载模型
    model = YOLO(args.model)

    # 开始训练（使用默认 project='runs/detect'）
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        cache=args.cache_type if args.cache else False,   # 支持 ram / disk
        name=args.name,             # 自定义运行名称
        exist_ok=True,              # 允许覆盖同名实验
        amp=True,                   # 开启混合精度训练
        # 默认保存路径
        verbose=False,
        plots=True,
    )

    print("\nTraining completed.")
    print("Results are saved in: runs/detect/train* (or runs/detect/<your-name> if --name is specified)")
    print("Best model  : runs/detect/<run>/weights/best.pt")
    print("Last model  : runs/detect/<run>/weights/last.pt")
    print("Metrics     : runs/detect/<run>/results.csv")
    print("Visualizations: confusion_matrix.png, results.png 等")


if __name__ == "__main__":
    main()