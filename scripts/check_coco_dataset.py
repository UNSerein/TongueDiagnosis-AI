import json
import os
import random
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

# ===== 1. 配置路径 =====
DATASET_ROOT = r"data/shezhenv3-coco"
SPLIT = "train"  # 可改为 val / test

IMG_DIR = os.path.join(DATASET_ROOT, SPLIT, "images")
ANN_DIR = os.path.join(DATASET_ROOT, SPLIT, "annotations")

# 自动找 COCO json
ann_files = [f for f in os.listdir(ANN_DIR) if f.endswith(".json")]
if len(ann_files) == 0:
    raise FileNotFoundError("No COCO json found in annotations folder")
ANN_PATH = os.path.join(ANN_DIR, ann_files[0])
print(f"Using annotation file: {ANN_PATH}")

# ===== 2. 读取 COCO 标注 =====
with open(ANN_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

print(f"Image count: {len(images)}")
print(f"Annotation count: {len(annotations)}")
print(f"Category count: {len(categories)}")

# ===== 3. 类别 ID → 名称 映射 =====
cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

print("\nCategory list:")
for cid, cname in sorted(cat_id_to_name.items()):
    print(f"{cid}: {cname}")

# ===== 4. 统计每个类别的 bbox 数量 =====
cat_count = defaultdict(int)
for ann in annotations:
    cat_count[ann["category_id"]] += 1

print("\nCategory annotation counts:")
for cid in sorted(cat_id_to_name.keys()):
    print(f"{cat_id_to_name[cid]:15s}: {cat_count.get(cid, 0)}")

# ===== 5. 随机可视化3张图片 =====
if len(images) > 0:
    sample_imgs = random.sample(images, min(3, len(images)))
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i, sample_img in enumerate(sample_imgs):
        img_path = os.path.join(IMG_DIR, sample_img["file_name"])
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Cannot read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 找到该图片的所有标注
        img_anns = [ann for ann in annotations if ann["image_id"] == sample_img["id"]]

        for ann in img_anns:
            x, y, w, h = map(int, ann["bbox"])
            cid = ann["category_id"]
            label = cat_id_to_name[cid]

            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img_rgb, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        axs[i].imshow(img_rgb)
        axs[i].set_title(f"Sample: {sample_img['file_name']}")
        axs[i].axis("off")

    plt.show()
else:
    print("No images found in dataset.")