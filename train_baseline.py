import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from dataset import TongueDataset

# =========================
# 1. 基本配置
# =========================
BATCH_SIZE = 2
NUM_EPOCHS = 3
NUM_CLASSES = 2      # 你现在是 2 个 demo 类
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. 加载数据
# =========================
dataset = TongueDataset("data/demo")
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================
# 3. 构建模型
# =========================
model = models.resnet18(pretrained=True)

# 🔑 修改最后一层（核心步骤）
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

# =========================
# 4. 损失函数 & 优化器
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# =========================
# 5. 训练循环
# =========================
model.train()

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

print("Training finished.")
