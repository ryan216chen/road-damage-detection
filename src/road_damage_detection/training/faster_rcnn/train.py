import argparse

import torch

from torch.optim import SGD

from torch.optim.lr_scheduler import (
    StepLR
)

from torch.utils.data import (
    DataLoader
)

from road_damage_detection.config.datasets import (
    add_dataset_argument,
    get_dataset_root
)

from road_damage_detection.config.faster_rcnn.runs import (
    get_run_dir
)

from road_damage_detection.config.faster_rcnn.settings import (
    FASTER_RCNN_EPOCHS,
    FASTER_RCNN_BATCH_SIZE,
    FASTER_RCNN_WORKERS,
    FASTER_RCNN_LEARNING_RATE,
    FASTER_RCNN_MOMENTUM,
    FASTER_RCNN_WEIGHT_DECAY,
    FASTER_RCNN_STEP_SIZE,
    FASTER_RCNN_GAMMA,
)

from road_damage_detection.datasets.faster_rcnn.dataset import (
    FasterRCNNDataset,
    collate_fn
)

from road_damage_detection.evaluation.faster_rcnn.validate import (
    evaluate
)

from road_damage_detection.models.faster_rcnn.model import (
    build_model
)


def train(
    dataset_name
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset_root = (
        get_dataset_root(
            dataset_name
        )
    )

    run_dir = (
        get_run_dir(
            dataset_name
        )
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_dataset = (
        FasterRCNNDataset(
            root=dataset_root,
            split="train"
        )
    )

    val_dataset = (
        FasterRCNNDataset(
            root=dataset_root,
            split="val"
        )
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=(
            FASTER_RCNN_BATCH_SIZE
        ),

        shuffle=True,

        num_workers=(
            FASTER_RCNN_WORKERS
        ),

        collate_fn=collate_fn
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

        collate_fn=collate_fn
    )

    model = build_model()

    model.to(
        device
    )

    parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = SGD(
        parameters,

        lr=(
            FASTER_RCNN_LEARNING_RATE
        ),

        momentum=(
            FASTER_RCNN_MOMENTUM
        ),

        weight_decay=(
            FASTER_RCNN_WEIGHT_DECAY
        )
    )

    scheduler = StepLR(
        optimizer,

        step_size=(
            FASTER_RCNN_STEP_SIZE
        ),

        gamma=(
            FASTER_RCNN_GAMMA
        )
    )

    best_map50 = -1.0

    for epoch in range(
        FASTER_RCNN_EPOCHS
    ):

        model.train()

        total_loss = 0.0

        for (
            images,
            targets
        ) in train_loader:

            images = [
                image.to(
                    device
                )
                for image
                in images
            ]

            targets = [
                {
                    key:
                        value.to(
                            device
                        )

                    for (
                        key,
                        value
                    ) in target.items()
                }

                for target
                in targets
            ]

            loss_dict = model(
                images,
                targets
            )

            loss = sum(
                loss_dict.values()
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        scheduler.step()

        average_loss = (
            total_loss
            / len(
                train_loader
            )
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

        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{FASTER_RCNN_EPOCHS}"
        )

        print(
            f"Loss = "
            f"{average_loss:.4f}"
        )

        print(
            f"mAP50 = "
            f"{map50:.4f}"
        )

        print(
            f"mAP50-95 = "
            f"{map5095:.4f}"
        )

        checkpoint = {
            "epoch": (
                epoch + 1
            ),

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "map50":
                map50,

            "map50_95":
                map5095,
        }

        torch.save(
            checkpoint,
            run_dir
            / "last.pt"
        )

        if map50 > best_map50:

            best_map50 = (
                map50
            )

            torch.save(
                checkpoint,
                run_dir
                / "best.pt"
            )

            print(
                "[INFO] "
                "New best model"
            )


def main():

    parser = (
        argparse.ArgumentParser()
    )

    add_dataset_argument(
        parser
    )

    args = (
        parser.parse_args()
    )

    train(
        dataset_name=(
            args.dataset
        )
    )


if __name__ == "__main__":
    main()