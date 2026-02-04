#该脚本用于冻结backbone，减少loss抖动
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from dataset import TongueDataset

# =========================
# 1. 基本配置
# =========================
BATCH_SIZE = 2
NUM_EPOCHS = 5
NUM_CLASSES = 2
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. 加载数据 & 划分训练 / 验证集
# =========================
dataset = TongueDataset("data/demo")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# 3. 构建模型（冻结 backbone）
# =========================
model = models.resnet18(pretrained=True)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 只训练最后的全连接层
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

# =========================
# 4. 损失函数 & 优化器
# =========================
criterion = nn.CrossEntropyLoss()

# 注意：只优化 fc 层参数
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

# =========================
# 5. 训练 + 验证
# =========================
for epoch in range(NUM_EPOCHS):
    # -------- train --------
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

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

    # -------- validation --------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f} | "
        f"Val Acc: {val_acc:.2f}"
    )

print("Training with frozen backbone finished.")
