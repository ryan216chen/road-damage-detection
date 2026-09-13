import argparse
from pathlib import Path

from ultralytics import YOLO

from road_damage_detection.config.paths import (
    PROJECT_ROOT,
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

from road_damage_detection.config.datasets import (
    add_dataset_argument,
    create_dataset_yaml
)

from road_damage_detection.training.mean_subtraction import (
    MeanSubtractionTrainer
)


TRAINING_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "training"
)


def get_run_name(
    dataset_name,
    mean_subtraction=False
):

    model_name = Path(
        YOLO_MODEL
    ).stem

    if mean_subtraction:
        return (
            f"{dataset_name}"
            f"_mean_subtraction_{model_name}"
        )

    return (
        f"{dataset_name}_{model_name}"
    )


def get_resume_checkpoint(
    run_name
):

    candidates = []

    for run_dir in TRAINING_ROOT.glob(
        f"{run_name}*"
    ):

        checkpoint = (
            run_dir
            / "weights"
            / "last.pt"
        )

        if checkpoint.exists():
            candidates.append(
                checkpoint
            )

    if not candidates:
        raise FileNotFoundError(
            f"No resumable checkpoint found for: "
            f"{run_name}"
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime
    )


def train(
    dataset_name,
    resume=False,
    mean_subtraction=False
):

    TRAINING_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    yaml_path = create_dataset_yaml(
        dataset_name
    )

    run_name = get_run_name(
        dataset_name,
        mean_subtraction
    )

    if resume:

        checkpoint = get_resume_checkpoint(
            run_name
        )

        print(
            f"[INFO] Resuming from: "
            f"{checkpoint}"
        )

        model = YOLO(
            str(checkpoint)
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

    add_dataset_argument(
        parser
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
