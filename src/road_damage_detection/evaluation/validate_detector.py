import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from road_damage_detection.config.datasets import (
    add_dataset_argument,
    create_dataset_yaml
)
from road_damage_detection.config.experiments import (
    add_experiment_arguments,
    resolve_experiment
)
from road_damage_detection.config.paths import (
    PROJECT_ROOT,
)
from road_damage_detection.config.runs import (
    VALIDATION_ROOT,
    get_checkpoint,
    get_run_dir_from_checkpoint,
    get_run_name
)
from road_damage_detection.config.settings import (
    TRAIN_IMAGE_SIZE,
    TRAIN_BATCH_SIZE,
    TRAIN_WORKERS,
    TRAIN_DEVICE,
)


def validate(
    dataset_name,
    weights=None,
    experiment_name=None,
    mean_subtraction=False
):

    experiment = resolve_experiment(
        experiment_name,
        mean_subtraction
    )

    run_name = get_run_name(
        dataset_name,
        experiment
    )

    if weights is None:

        weights_path = get_checkpoint(
            run_name,
            filename="best.pt"
        )

    else:

        weights_path = Path(
            weights
        )

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

    actual_run_name = (
        get_run_dir_from_checkpoint(
            weights_path
        ).name
    )

    VALIDATION_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    output_dir = (
        VALIDATION_ROOT
        / actual_run_name
    )

    if output_dir.exists():

        print(
            f"[INFO] Removing old validation results: "
            f"{output_dir}"
        )

        shutil.rmtree(
            output_dir
        )

    yaml_path = create_dataset_yaml(
        dataset_name
    )

    print(
        f"[INFO] Dataset: "
        f"{dataset_name}"
    )

    print(
        f"[INFO] Experiment: "
        f"{experiment.name}"
    )

    print(
        f"[INFO] Weights: "
        f"{weights_path}"
    )

    model = YOLO(
        str(weights_path)
    )

    val_args = {
        "data": str(yaml_path),
        "split": "val",
        "imgsz": TRAIN_IMAGE_SIZE,
        "batch": TRAIN_BATCH_SIZE,
        "workers": TRAIN_WORKERS,
        "device": TRAIN_DEVICE,
        "project": str(VALIDATION_ROOT),
        "name": actual_run_name,
        "plots": True,
        "exist_ok": True
    }

    if experiment.validator is not None:
        val_args["validator"] = (
            experiment.validator
        )

    metrics = model.val(
        **val_args
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

    add_dataset_argument(
        parser
    )

    add_experiment_arguments(
        parser
    )

    parser.add_argument(
        "--weights",
        default=None
    )

    args = parser.parse_args()

    validate(
        dataset_name=args.dataset,
        weights=args.weights,
        experiment_name=args.experiment,
        mean_subtraction=args.mean_subtraction
    )


if __name__ == "__main__":
    main()
