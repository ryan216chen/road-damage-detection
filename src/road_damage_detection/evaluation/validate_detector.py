import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from road_damage_detection.config.paths import (
    PROJECT_ROOT,
    SEGMENTATION_ROOT,
    HISTOGRAM_EQUALIZATION_ROOT,
    HISTOGRAM_MATCHING_ROOT,
)

from road_damage_detection.config.settings import (
    YOLO_MODEL,
    TRAIN_IMAGE_SIZE,
    TRAIN_BATCH_SIZE,
    TRAIN_WORKERS,
    TRAIN_DEVICE,
)

from road_damage_detection.config.datasets import (
    get_dataset_root,
    add_dataset_argument
)

TRAINING_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "training"
)

VALIDATION_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "validation"
)



def validate(
    dataset_name,
    weights=None,
):

    dataset_root = get_dataset_root(dataset_name)

    # 如果沒有指定 weights，自動找 best.pt
    if weights is None:

        model_name = Path(
            YOLO_MODEL
        ).stem

        weights_path = (
            TRAINING_ROOT
            / f"{dataset_name}_{model_name}"
            / "weights"
            / "best.pt"
        )

    else:

        weights_path = Path(
            weights
        )

        # 相對路徑 → 以專案根目錄為基準
        if not weights_path.is_absolute():

            weights_path = (
                PROJECT_ROOT
                / weights_path
            )

    if not weights_path.exists():

        raise FileNotFoundError(
            f"Weights not found: "
            f"{weights_path}"
        )

    VALIDATION_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # 如果之前的 validation 結果存在，就先刪掉
    output_dir = (
        VALIDATION_ROOT
        / dataset_name
    )

    if output_dir.exists():

        print(
            f"[INFO] Removing old validation results: "
            f"{output_dir}"
        )

        shutil.rmtree(
            output_dir
        )

    yaml_path = (
        VALIDATION_ROOT
        / f"{dataset_name}.yaml"
    )

    content = f"""
path: "{dataset_root.as_posix()}"

train: images/train
val: images/val

names:
  0: D00
  1: D10
  2: D20
  3: D40
"""

    yaml_path.write_text(
        content.strip(),
        encoding="utf-8"
    )

    print(
        f"[INFO] Dataset: "
        f"{dataset_name}"
    )

    print(
        f"[INFO] Weights: "
        f"{weights_path}"
    )

    model = YOLO(
        str(weights_path)
    )

    metrics = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=TRAIN_IMAGE_SIZE,
        batch=TRAIN_BATCH_SIZE,
        workers=TRAIN_WORKERS,
        device=TRAIN_DEVICE,
        project=str(VALIDATION_ROOT),
        name=dataset_name,
        plots=True,
        exist_ok=True,
    )

    print()
    print("===== Overall Metrics =====")

    print(
        f"Precision   = {metrics.box.mp:.4f}"
    )

    print(
        f"Recall      = {metrics.box.mr:.4f}"
    )

    print(
        f"mAP50       = {metrics.box.map50:.4f}"
    )

    print(
        f"mAP50-95    = {metrics.box.map:.4f}"
    )

    print()
    print("===== Per-Class Metrics =====")

    for i, class_index in enumerate(
        metrics.box.ap_class_index
    ):

        precision, recall, ap50, ap = (
            metrics.box.class_result(i)
        )

        class_name = model.names[
            int(class_index)
        ]

        print(
            f"{class_name}: "
            f"P = {precision:.4f}, "
            f"R = {recall:.4f}, "
            f"AP50 = {ap50:.4f}, "
            f"AP50-95 = {ap:.4f}"
        )


def main():

    parser = argparse.ArgumentParser()

    add_dataset_argument(parser)

    parser.add_argument(
        "--weights",
        default=None
    )

    args = parser.parse_args()

    validate(
        args.dataset,
        args.weights,
    )


if __name__ == "__main__":
    main()