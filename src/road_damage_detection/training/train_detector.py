import argparse

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

from road_damage_detection.training.mean_subtraction import (
    MeanSubtractionTrainer
)

from road_damage_detection.config.datasets import (
    get_dataset_root,
    add_dataset_argument,
    create_dataset_yaml
)



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

    yaml_path = create_dataset_yaml(dataset_name)
    
    TRAINING_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    create_dataset_yaml(
        dataset_name,
        yaml_path
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

    add_dataset_argument(parser)

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