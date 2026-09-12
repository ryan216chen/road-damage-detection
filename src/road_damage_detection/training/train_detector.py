import argparse

from ultralytics import YOLO

from road_damage_detection.config.paths import (
    SEGMENTATION_ROOT,
    HISTOGRAM_EQUALIZATION_ROOT,
    HISTOGRAM_MATCHING_ROOT,
    PROJECT_ROOT,
    ITERATIVE_ROOT
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

from road_damage_detection.training.mean_subtraction import (
    MeanSubtractionTrainer
)


DATASET_ROOTS = {
    "baseline": SEGMENTATION_ROOT,
    "equalized": HISTOGRAM_EQUALIZATION_ROOT,
    "matched": HISTOGRAM_MATCHING_ROOT,
    "iterative": ITERATIVE_ROOT
}


TRAINING_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "training"
)


def train(
    dataset_name,
    resume=False,
    mean_subtraction=False
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

    if mean_subtraction:

        run_name = (
            f"{dataset_name}"
            "_mean_subtraction_yolo11m"
        )

    else:

        run_name = (
            f"{dataset_name}_yolo11m"
        )

    if resume:

        checkpoint = (
            TRAINING_ROOT
            / run_name
            / "weights"
            / "last.pt"
        )

        model = YOLO(
            checkpoint
        )

        if mean_subtraction:

            model.train(
                resume=True,
                trainer=MeanSubtractionTrainer
            )

        else:

            model.train(
                resume=True
            )

        return

    model = YOLO(
        YOLO_MODEL
    )

    train_args = {
        "data": str(yaml_path),
        "epochs": TRAIN_EPOCHS,
        "imgsz": TRAIN_IMAGE_SIZE,
        "batch": TRAIN_BATCH_SIZE,
        "workers": TRAIN_WORKERS,
        "device": TRAIN_DEVICE,
        "close_mosaic": TRAIN_CLOSE_MOSAIC,
        "seed": RANDOM_SEED,
        "deterministic": TRAIN_DETERMINISTIC,
        "project": str(TRAINING_ROOT),
        "name": run_name
    }

    if mean_subtraction:

        train_args["trainer"] = (
            MeanSubtractionTrainer
        )

    model.train(
        **train_args
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=[
            "baseline",
            "equalized",
            "matched",
            "iterative"
        ],
        required=True
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    parser.add_argument(
        "--mean-subtraction",
        action="store_true"
    )

    args = parser.parse_args()

    train(
        args.dataset,
        args.resume,
        args.mean_subtraction
    )


if __name__ == "__main__":
    main()