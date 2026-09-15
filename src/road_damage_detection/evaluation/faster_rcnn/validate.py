import argparse
from pathlib import Path

import torch

from torch.utils.data import (
    DataLoader
)

from torchmetrics.detection.mean_ap import (
    MeanAveragePrecision
)

from road_damage_detection.config.datasets import (
    add_dataset_argument,
    get_dataset_root
)

from road_damage_detection.config.faster_rcnn.runs import (
    get_checkpoint
)

from road_damage_detection.config.faster_rcnn.settings import (
    FASTER_RCNN_BATCH_SIZE,
    FASTER_RCNN_WORKERS
)

from road_damage_detection.datasets.faster_rcnn.dataset import (
    FasterRCNNDataset,
    collate_fn
)

from road_damage_detection.models.faster_rcnn.model import (
    build_model
)


@torch.inference_mode()
def evaluate(
    model,
    data_loader,
    device
):

    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox"
    )

    for (
        images,
        targets
    ) in data_loader:

        images = [
            image.to(
                device
            )
            for image
            in images
        ]

        outputs = model(
            images
        )

        predictions = []

        ground_truths = []

        for (
            output,
            target
        ) in zip(
            outputs,
            targets
        ):

            prediction = {
                "boxes": (
                    output["boxes"]
                    .detach()
                    .cpu()
                ),

                "scores": (
                    output["scores"]
                    .detach()
                    .cpu()
                ),

                "labels": (
                    output["labels"]
                    .detach()
                    .cpu()
                ),
            }

            ground_truth = {
                "boxes": (
                    target["boxes"]
                    .detach()
                    .cpu()
                ),

                "labels": (
                    target["labels"]
                    .detach()
                    .cpu()
                ),
            }

            predictions.append(
                prediction
            )

            ground_truths.append(
                ground_truth
            )

        metric.update(
            predictions,
            ground_truths
        )

    results = metric.compute()

    return results


def validate(
    dataset_name,
    weights=None
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[INFO] Device: "
        f"{device}"
    )

    dataset_root = (
        get_dataset_root(
            dataset_name
        )
    )

    print(
        f"[INFO] Dataset: "
        f"{dataset_name}"
    )

    print(
        f"[INFO] Dataset root: "
        f"{dataset_root}"
    )

    val_dataset = (
        FasterRCNNDataset(
            root=dataset_root,
            split="val"
        )
    )

    print(
        f"[INFO] Validation images: "
        f"{len(val_dataset)}"
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=(
            FASTER_RCNN_BATCH_SIZE
        ),

        shuffle=False,

        num_workers=(
            FASTER_RCNN_WORKERS
        ),

        collate_fn=(
            collate_fn
        ),

        pin_memory=(
            device.type == "cuda"
        )
    )

    model = build_model()

    if weights is None:

        checkpoint_path = (
            get_checkpoint(
                dataset_name,
                filename="best.pt"
            )
        )

    else:

        checkpoint_path = Path(
            weights
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

    print(
        f"[INFO] Checkpoint: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        model_state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model_state_dict = (
            checkpoint
        )

    model.load_state_dict(
        model_state_dict
    )

    model.to(
        device
    )

    results = evaluate(
        model,
        val_loader,
        device
    )

    map50 = (
        results[
            "map_50"
        ].item()
    )

    map5095 = (
        results[
            "map"
        ].item()
    )

    print()
    print(
        "===== Faster R-CNN Validation ====="
    )

    print(
        f"mAP50    = "
        f"{map50:.4f}"
    )

    print(
        f"mAP50-95 = "
        f"{map5095:.4f}"
    )


def main():

    parser = (
        argparse.ArgumentParser()
    )

    add_dataset_argument(
        parser
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None
    )

    args = (
        parser.parse_args()
    )

    validate(
        dataset_name=(
            args.dataset
        ),

        weights=(
            args.weights
        )
    )


if __name__ == "__main__":
    main()