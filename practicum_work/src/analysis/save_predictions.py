import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from mmseg.apis import init_model, inference_model


CONFIG = "configs/practicum/segformer_b2_class_weights.py"
CHECKPOINT = "work_dirs/practicum/segformer_b2_class_weights/best_mDice_iter_3500.pth"

IMG_DIR = Path("data/train_dataset_for_students_clean/img/test")
MASK_DIR = Path("data/train_dataset_for_students_clean/labels/test")

OUT_DIR = Path("practicum_work/supplementary/viz/predictions")
BEST_DIR = OUT_DIR / "best"
WORST_DIR = OUT_DIR / "worst"

BEST_DIR.mkdir(parents=True, exist_ok=True)
WORST_DIR.mkdir(parents=True, exist_ok=True)


def find_mask_for_image(img_path: Path) -> Path:
    candidates = list(MASK_DIR.glob(img_path.stem + ".*"))
    if not candidates:
        raise FileNotFoundError(f"Mask not found for image: {img_path.name}")
    return candidates[0]


def dice_score(pred, gt, cls):
    pred_cls = pred == cls
    gt_cls = gt == cls

    inter = np.logical_and(pred_cls, gt_cls).sum()
    union = pred_cls.sum() + gt_cls.sum()

    if union == 0:
        return 1.0

    return 2 * inter / union


def mean_dice_no_background(pred, gt):
    dice1 = dice_score(pred, gt, 1)
    dice2 = dice_score(pred, gt, 2)
    return (dice1 + dice2) / 2


def save_comparison(img_path, gt, pred, score, out_path):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, vmin=0, vmax=2)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, vmin=0, vmax=2)
    plt.title(f"Prediction\nmDice={score:.3f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


model = init_model(CONFIG, CHECKPOINT, device="cuda")

results = []

image_paths = sorted(list(IMG_DIR.glob("*")))

for img_path in tqdm(image_paths):
    mask_path = find_mask_for_image(img_path)

    gt = cv2.imread(str(mask_path), 0)

    if gt is None:
        print(f"Could not read mask: {mask_path}")
        continue

    result = inference_model(model, str(img_path))
    pred = result.pred_sem_seg.data.cpu().numpy()[0]

    score = mean_dice_no_background(pred, gt)

    results.append({
        "name": img_path.name,
        "img_path": img_path,
        "mask_path": mask_path,
        "score": score,
        "gt": gt,
        "pred": pred,
    })


results = sorted(results, key=lambda x: x["score"])

rows = []

for i, item in enumerate(results[:5]):
    out_path = WORST_DIR / f"worst_{i}_{item['score']:.3f}_{item['name']}.png"
    save_comparison(item["img_path"], item["gt"], item["pred"], item["score"], out_path)
    rows.append({"type": "worst", "name": item["name"], "score": item["score"], "path": str(out_path)})

for i, item in enumerate(results[-5:][::-1]):
    out_path = BEST_DIR / f"best_{i}_{item['score']:.3f}_{item['name']}.png"
    save_comparison(item["img_path"], item["gt"], item["pred"], item["score"], out_path)
    rows.append({"type": "best", "name": item["name"], "score": item["score"], "path": str(out_path)})


df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "prediction_examples.csv", index=False)

print("Saved examples to:", OUT_DIR)
print(df)
