import argparse
import subprocess
from pathlib import Path

from clearml import Task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to mmseg config")
    parser.add_argument("--task-name", required=True, help="ClearML task name")
    parser.add_argument(
        "--project-name",
        default="MMSegmentation Practicum",
        help="ClearML project name",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Optional mmseg work_dir",
    )
    args = parser.parse_args()

    task = Task.init(
        project_name=args.project_name,
        task_name=args.task_name,
    )

    task.connect({
        "config": args.config,
        "work_dir": args.work_dir,
    })

    cmd = [
        "python",
        "tools/train.py",
        args.config,
    ]

    if args.work_dir is not None:
        cmd += ["--work-dir", args.work_dir]

    print("Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
