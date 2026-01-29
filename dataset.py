import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TongueDataset(Dataset):
    """
    自定义舌象数据集类
    数据组织形式：
    root_dir/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img3.jpg
            img4.jpg
    """

    def __init__(self, root_dir):
        """
        参数：
        root_dir: 数据集根目录
        """
        self.root_dir = root_dir
        self.samples = []          # 存储 (图片路径, 类别标签)
        self.class_to_idx = {}     # 类名 -> 数字标签

        # 读取所有类别文件夹
        classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_dir = os.path.join(root_dir, cls)

            # 遍历类别文件夹下的所有图片
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), idx)
                    )

        # 图像预处理：Resize + 转 Tensor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        根据索引返回一条样本
        返回：
        image: Tensor [3, 224, 224]
        label: int
        """
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
