import json
import os
import random
import cv2
import matplotlib.pyplot as plt

# ===== 1. 配置路径 =====
DATASET_ROOT = r"D:\ADMIN\\TongueDiagnosis-AI\shezhenv3-coco"
SPLIT = "train"   # 可改为 val / test

IMG_DIR = os.path.join(DATASET_ROOT, SPLIT, "images")
ANN_DIR = os.path.join(DATASET_ROOT, SPLIT, "annotations")

# 自动找 COCO json
ann_files = [f for f in os.listdir(ANN_DIR) if f.endswith(".json")]
assert len(ann_files) > 0, "annotations 文件夹中未找到 COCO json 文件"
ANN_PATH = os.path.join(ANN_DIR, ann_files[0])

print(f"使用标注文件: {ANN_PATH}")

# ===== 2. 读取 COCO 标注 =====
with open(ANN_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

print(f"图片数量: {len(images)}")
print(f"标注框数量: {len(annotations)}")
print(f"类别数量: {len(categories)}")

# ===== 3. 类别 ID → 名称 映射 =====
cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

print("\n类别列表：")
for cid, cname in cat_id_to_name.items():
    print(f"{cid}: {cname}")

# ===== 4. 统计每个类别的 bbox 数量 =====
from collections import defaultdict

cat_count = defaultdict(int)
for ann in annotations:
    cat_count[ann["category_id"]] += 1

print("\n各类别标注数量：")
for cid in sorted(cat_id_to_name.keys()):
    print(f"{cat_id_to_name[cid]:15s}: {cat_count[cid]}")

# ===== 5. 随机可视化一张图片 =====
sample_img = random.choice(images)
img_path = os.path.join(IMG_DIR, sample_img["file_name"])

img = cv2.imread(img_path)
assert img is not None, f"无法读取图像: {img_path}"

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 找到该图片的所有标注
img_anns = [ann for ann in annotations if ann["image_id"] == sample_img["id"]]

for ann in img_anns:
    x, y, w, h = map(int, ann["bbox"])
    cid = ann["category_id"]
    label = cat_id_to_name[cid]

    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(
        img_rgb,
        label,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )

plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title(f"Sample Image: {sample_img['file_name']}")
plt.axis("off")
plt.show()
