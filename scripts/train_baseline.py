import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from dataset import TongueDataset
from datetime import datetime

# =========================
# 1. 基本配置
# =========================
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_CLASSES = 21  # 正式舌象数据集

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. 数据集加载
# =========================
dataset_root = "data/shezhenv3-coco/train/images"  # 图片根目录
dataset = TongueDataset(dataset_root)

# 划分训练/验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# =========================
# 3.模型构建
# =========================
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================
# 4. 创建实验结果目录
# =========================
experiment_dir = "experiments/exp1_yolov8n_cpu_ep5/results"
os.makedirs(experiment_dir, exist_ok=True)

# =========================
# 5. 训练循环 + 结果记录
# =========================
metrics_file = os.path.join(experiment_dir, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
with open(metrics_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

for epoch in range(NUM_EPOCHS):
    # -------- train --------
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss /= len(train_loader)

    # -------- validation --------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    val_loss /= len(val_loader)

    # -------- 打印 & 保存 --------
    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

    # 写入 CSV
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])

# =========================
# 6. 保存模型权重
# =========================
weights_path = os.path.join(experiment_dir, "resnet18_cpu_baseline.pt")
torch.save(model.state_dict(), weights_path)
print(f"训练完成，模型权重保存至: {weights_path}")
print(f"训练指标保存至: {metrics_file}")
