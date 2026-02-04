import torch
from torch.utils.data import DataLoader
from torchvision import models
from dataset import TongueDataset


# 1. 加载数据集
# 这里使用的是 demo 数据，仅用于流程验证
dataset = TongueDataset("data/demo")

# DataLoader 用于批量加载数据
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

# 2. 加载预训练 ResNet18
# 使用 ImageNet 预训练权重
model = models.resnet18(pretrained=True)

# 设置为评估模式（不启用 dropout / BN 更新）
model.eval()


# 3. 前向推理（不计算梯度）
images, labels = next(iter(loader))

with torch.no_grad():
    outputs = model(images)

# 4. 打印结果
print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
