from road_damage_detection.config.paths import (
    PROJECT_ROOT 
)

FASTER_RCNN_TRAININIG_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "training"
    / "faster_rcnn"
)

def get_run_name(
    dataset_name
):

    return (
        f"{dataset_name}_"
        f"resnet50_fpn_v2"
    )

def get_run_dir(
    dataset_name
):

    return (
        FASTER_RCNN_TRAININIG_ROOT
        / get_run_name(
            dataset_name 
        )
    )


def get_checkpoint(
    dataset_name,
    filename = "best.pt"
):

    checkpoint = (
        get_run_dir(
            dataset_name 
        )
        / filename 
    )

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"Faster R-CNN checkpoint"
            f"not found : {checkpoint}" 
        )

    return checkpoint 