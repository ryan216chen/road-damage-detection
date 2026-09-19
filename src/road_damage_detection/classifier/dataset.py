from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from ultralytics.data.augment import (
    classify_augmentations,
    classify_transforms
)


MAX_WORKERS = 8


def build_sample(
    image_path: Path,
    image_dir: Path,
    label_dir: Path
):

    relative_path = (
        image_path
        .relative_to(image_dir)
    )

    label_path = (
        label_dir
        / relative_path
    ).with_suffix(
        ".txt"
    )

    if not label_path.exists():

        raise FileNotFoundError(
            f"Label not found: {label_path}"
        )

    has_damage = bool(
        label_path.read_text(
            encoding="utf-8"
        ).strip()
    )

    class_id = (
        1
        if has_damage
        else 0
    )

    return (
        str(image_path),
        class_id
    )


class IterativeClassificationDataset(
    Dataset
):

    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        args,
        augment: bool = False,
        prefix: str = ""
    ):

        self.image_dir = Path(
            image_dir
        )

        self.label_dir = Path(
            label_dir
        )

        image_paths = sorted(
            self.image_dir.rglob(
                "*.jpg"
            )
        )

        worker = partial(
            build_sample,
            image_dir=self.image_dir,
            label_dir=self.label_dir
        )

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            self.samples = list(
                tqdm(
                    executor.map(
                        worker,
                        image_paths
                    ),
                    total=len(image_paths),
                    desc=f"Loading {prefix}"
                )
            )

        if augment:

            self.torch_transforms = (
                classify_augmentations(
                    size=args.imgsz,
                    scale=(
                        1.0 - args.scale,
                        1.0
                    ),
                    hflip=args.fliplr,
                    vflip=args.flipud,
                    erasing=args.erasing,
                    auto_augment=args.auto_augment,
                    hsv_h=args.hsv_h,
                    hsv_s=args.hsv_s,
                    hsv_v=args.hsv_v
                )
            )

        else:

            self.torch_transforms = (
                classify_transforms(
                    size=args.imgsz
                )
            )

    def __len__(
        self
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int
    ):

        (
            image_path,
            class_id
        ) = self.samples[
            index
        ]

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise RuntimeError(
                f"Cannot read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            image
        )

        image = self.torch_transforms(
            image
        )

        return {
            "img": image,
            "cls": class_id
        }
