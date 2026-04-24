from __future__ import annotations

"""Шаблон скрипта для сохранения предсказаний модели.

Здесь позже можно подключить inferencer / runner API mmsegmentation.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Template script. Integrate it with your mmseg inference pipeline.')
    print('Config:     ', args.config)
    print('Checkpoint: ', args.checkpoint)
    print('Input dir:  ', args.input_dir)
    print('Output dir: ', args.output_dir)


if __name__ == '__main__':
    main()
