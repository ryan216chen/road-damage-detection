import argparse

from ultralytics import YOLO

from road_damage_detection.config.datasets import (
    add_dataset_argument,
    create_dataset_yaml
)

from road_damage_detection.config.experiments import (
    add_experiment_arguments,
    resolve_experiment
)

from road_damage_detection.config.runs import (
    TRAINING_ROOT,
    get_checkpoint,
    get_run_name
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


def train(
    dataset_name,
    resume=False,
    experiment_name=None,
    mean_subtraction=False
):

    TRAINING_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    yaml_path = create_dataset_yaml(
        dataset_name
    )

    experiment = resolve_experiment(
        experiment_name,
        mean_subtraction
    )

    run_name = get_run_name(
        dataset_name,
        experiment
    )

    if resume:

        checkpoint = get_checkpoint(
            run_name,
            filename="last.pt"
        )

        print(
            f"[INFO] Resuming from: "
            f"{checkpoint}"
        )

        model = YOLO(
            str(checkpoint)
        )

        resume_args = {
            "resume": True
        }

        if experiment.trainer is not None:

            resume_args["trainer"] = (
                experiment.trainer
            )

        model.train(
            **resume_args
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

    if experiment.trainer is not None:

        train_args["trainer"] = (
            experiment.trainer
        )

    model.train(
        **train_args
    )


def main():

    parser = argparse.ArgumentParser()

    add_dataset_argument(
        parser
    )

    add_experiment_arguments(
        parser
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        resume=args.resume,
        experiment_name=args.experiment,
        mean_subtraction=args.mean_subtraction
    )


if __name__ == "__main__":
    main()