from pathlib import Path

from road_damage_detection.config.paths import (
    PROJECT_ROOT,
)

from road_damage_detection.config.settings import (
    YOLO_MODEL,
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

PREDICTION_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "prediction"
)


def get_run_name(
    dataset_name,
    experiment
):

    model_name = Path(
        YOLO_MODEL
    ).stem

    if experiment.run_suffix is None:
        return (
            f"{dataset_name}_{model_name}"
        )

    return (
        f"{dataset_name}_"
        f"{experiment.run_suffix}_"
        f"{model_name}"
    )


def find_latest_run(
    run_name
):

    candidates = []

    for run_dir in TRAINING_ROOT.glob(
        f"{run_name}*"
    ):

        if run_dir.is_dir():
            candidates.append(
                run_dir
            )

    if not candidates:
        raise FileNotFoundError(
            f"Training run not found: "
            f"{run_name}"
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime
    )


def get_checkpoint(
    run_name,
    filename="best.pt"
):

    run_dir = find_latest_run(
        run_name
    )

    checkpoint = (
        run_dir
        / "weights"
        / filename
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint}"
        )

    return checkpoint


def get_run_dir_from_checkpoint(
    checkpoint_path
):

    return (
        Path(checkpoint_path)
        .parent
        .parent
    )
