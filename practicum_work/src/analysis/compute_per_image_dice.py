from __future__ import annotations

"""Подсчёт Dice score по каждому изображению (бинарный случай)."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def dice_score_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    intersection = np.logical_and(pred == 1, gt == 1).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', required=True)
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--out-csv', required=True)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for pred_path in sorted([p for p in pred_dir.iterdir() if p.is_file()]):
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            print(f'[WARN] GT not found for {pred_path.name}')
            continue

        pred = np.array(Image.open(pred_path))
        gt = np.array(Image.open(gt_path))
        if pred.ndim == 3:
            pred = pred[..., 0]
        if gt.ndim == 3:
            gt = gt[..., 0]

        rows.append({'file_name': pred_path.name, 'dice': dice_score_binary(pred, gt)})

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file_name', 'dice'])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        mean_dice = sum(r['dice'] for r in rows) / len(rows)
        print(f'Mean Dice: {mean_dice:.6f}')
    print(f'Saved report to: {out_csv}')


if __name__ == '__main__':
    main()
