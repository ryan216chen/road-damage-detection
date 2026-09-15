from pathlib import Path

import torch

from PIL import Image

from torch.utils.data import Dataset

from torchvision.transforms.functional import (
    pil_to_tensor
)


class FasterRCNNDataset(
    Dataset
):

    def __init__(
        self,
        root,
        split
    ):

        self.root = Path(
            root
        )

        self.image_root = (
            self.root
            / "images"
            / split
        )

        self.label_root = (
            self.root
            / "labels"
            / split
        )

        self.image_paths = sorted(
            self.image_root.rglob(
                "*.jpg"
            )
        )

        if not self.image_paths:

            raise FileNotFoundError(
                f"No images found: "
                f"{self.image_root}"
            )

    def __len__(
        self
    ):

        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        index
    ):

        image_path = (
            self.image_paths[
                index
            ]
        )

        image = (
            Image
            .open(
                image_path
            )
            .convert(
                "RGB"
            )
        )

        width, height = (
            image.size
        )

        relative_path = (
            image_path.relative_to(
                self.image_root
            )
        )

        label_path = (
            self.label_root
            / relative_path
        ).with_suffix(
            ".txt"
        )

        boxes = []

        labels = []

        if label_path.exists():

            lines = (
                label_path
                .read_text(
                    encoding="utf-8"
                )
                .splitlines()
            )

            for line in lines:

                if not line.strip():
                    continue

                (
                    class_id,
                    x_center,
                    y_center,
                    box_width,
                    box_height
                ) = map(
                    float,
                    line.split()
                )

                x_center *= width

                y_center *= height

                box_width *= width

                box_height *= height

                x1 = (
                    x_center
                    - box_width / 2
                )

                y1 = (
                    y_center
                    - box_height / 2
                )

                x2 = (
                    x_center
                    + box_width / 2
                )

                y2 = (
                    y_center
                    + box_height / 2
                )

                boxes.append([
                    x1,
                    y1,
                    x2,
                    y2
                ])

                labels.append(
                    int(class_id)
                    + 1
                )

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        ).reshape(
            -1,
            4
        )

        labels = torch.tensor(
            labels,
            dtype=torch.int64
        )

        image = (
            pil_to_tensor(
                image
            )
            .float()
            / 255.0
        )

        target = {
            "boxes": boxes,

            "labels": labels,

            "image_id": torch.tensor(
                index,
                dtype=torch.int64
            )
        }

        return (
            image,
            target
        )


def collate_fn(
    batch
):

    return tuple(
        zip(
            *batch
        )
    )