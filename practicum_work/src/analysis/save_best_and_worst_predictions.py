from __future__ import annotations

"""Сохранение лучших и худших примеров по Dice score."""

import argparse
import csv
from pathlib import Path
from shutil import copy2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores-csv', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--pred-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()

    scores_csv = Path(args.scores_csv)
    images_dir = Path(args.images_dir)
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    best_dir = out_dir / 'best'
    worst_dir = out_dir / 'worst'
    best_dir.mkdir(parents=True, exist_ok=True)
    worst_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with scores_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['dice'] = float(row['dice'])
            rows.append(row)

    rows_sorted = sorted(rows, key=lambda x: x['dice'])
    worst = rows_sorted[:args.top_k]
    best = rows_sorted[-args.top_k:]

    def copy_triplet(items, dst_dir: Path) -> None:
        for row in items:
            file_name = row['file_name']
            stem = Path(file_name).stem
            for src_dir, suffix in [(images_dir, '_image'), (gt_dir, '_gt'), (pred_dir, '_pred')]:
                src = src_dir / file_name
                if src.exists():
                    dst = dst_dir / f'{stem}{suffix}{src.suffix}'
                    copy2(src, dst)

    copy_triplet(best, best_dir)
    copy_triplet(worst, worst_dir)

    print(f'Saved best examples to:  {best_dir}')
    print(f'Saved worst examples to: {worst_dir}')


if __name__ == '__main__':
    main()
