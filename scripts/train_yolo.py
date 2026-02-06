import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script for Tongue Diagnosis")
    parser.add_argument('--model', type=str, default='models/yolov8n.pt',
                        help='Path to model weights (e.g., yolov8n.pt, yolov8s.pt)')
    parser.add_argument('--data', type=str, default='data/dataset.yaml',
                        help='Path to dataset.yaml')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch', type=str, default='-1',
                        help='Batch size: number (e.g. 16), -1 for AutoBatch, 0.8 for 80% memory')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (pixels)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device: "cpu" or GPU index like "0"')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loader workers')
    parser.add_argument('--cache', action='store_true',
                        help='Cache images to RAM/disk for faster loading')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom name for this run (will appear in runs/detect/<name>)')
    
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载模型
    model = YOLO(args.model)

    # 训练
    print(f"Starting training with:")
    print(f"  Model:     {args.model}")
    print(f"  Data:      {args.data}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch} (auto if -1)")
    print(f"  Image size:{args.imgsz}")
    print(f"  Device:    {args.device}")
    print(f"  Workers:   {args.workers}")
    print(f"  Cache:     {args.cache}")
    if args.name:
        print(f"  Run name:  {args.name}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        name=args.name,           # 如果指定 name，会保存到 runs/detect/<name>
        exist_ok=True,            # 允许覆盖同名实验
        project='runs/detect',    # 默认项目目录（可改，但通常不用动）
        amp=True,                 # 混合精度加速
    )

    print("\nTraining completed.")
    print("Results are saved in: runs/detect/train* (or the custom name you specified)")
    print("Best weights: runs/detect/train*/weights/best.pt")
    print("Last weights: runs/detect/train*/weights/last.pt")


if __name__ == "__main__":
    main()