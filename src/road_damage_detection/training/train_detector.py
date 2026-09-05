import argparse

from ultralytics import YOLO

from road_damage_detection.config.paths import (
    SEGMENTATION_ROOT,
    HISTOGRAM_EQUALIZATION_ROOT,
    HISTOGRAM_MATCHING_ROOT,
    PROJECT_ROOT
)

from road_damage_detection.config.settings import (
    YOLO_MODEL,
    TRAIN_EPOCHS,
    TRAIN_IMAGE_SIZE,
    TRAIN_BATCH_SIZE,
    TRAIN_WORKERS,
    TRAIN_DEVICE,
    TRAIN_CLOSE_MOSAIC,
    TRAIN_DETERMINISTIC,
    RANDOM_SEED
)


DATASET_ROOTS = {
    "baseline": SEGMENTATION_ROOT,
    "equalized": HISTOGRAM_EQUALIZATION_ROOT,
    "matched": HISTOGRAM_MATCHING_ROOT,
}


TRAINING_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "training"
)


def train(
    dataset_name
):

    dataset_root = (
        DATASET_ROOTS[
            dataset_name
        ]
    )

    yaml_path = (
        TRAINING_ROOT
        / f"{dataset_name}.yaml"
    )

    TRAINING_ROOT.mkdir(
        parents=True,
        exist_ok=True
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

    model = YOLO(
        YOLO_MODEL
    )

    model.train(
        data=str(yaml_path),
        epochs=TRAIN_EPOCHS,
        imgsz=TRAIN_IMAGE_SIZE,
        batch=TRAIN_BATCH_SIZE,
        workers=TRAIN_WORKERS,
        device=TRAIN_DEVICE,
        close_mosaic=TRAIN_CLOSE_MOSAIC,
        seed=RANDOM_SEED,
        deterministic=TRAIN_DETERMINISTIC,
        project=str(TRAINING_ROOT),
        name=f"{dataset_name}_yolo11m"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=[
            "baseline",
            "equalized",
            "matched"
        ],
        required=True
    )

    args = parser.parse_args()

    train(
        args.dataset
    )


if __name__ == "__main__":
    main()