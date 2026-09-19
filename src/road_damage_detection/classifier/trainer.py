from copy import copy
from pathlib import Path

from ultralytics.models.yolo.classify.train import (
    ClassificationTrainer
)

from ultralytics.models.yolo.classify.val import (
    ClassificationValidator
)

from road_damage_detection.classifier.dataset import (
    IterativeClassificationDataset
)

from road_damage_detection.config.paths import (
    ITERATIVE_IMAGE_ROOT,
    ITERATIVE_LABEL_ROOT
)


class IterativeClassificationValidator(
    ClassificationValidator
):

    def build_dataset(
        self,
        img_path
    ):

        split = Path(
            img_path
        ).name

        return (
            IterativeClassificationDataset(
                image_dir=Path(img_path),
                label_dir=(
                    ITERATIVE_LABEL_ROOT
                    / split
                ),
                args=self.args,
                augment=False,
                prefix=f"val-{split}"
            )
        )


class IterativeClassificationTrainer(
    ClassificationTrainer
):

    def get_dataset(
        self
    ):

        return {
            "train": str(
                ITERATIVE_IMAGE_ROOT
                / "train"
            ),
            "val": str(
                ITERATIVE_IMAGE_ROOT
                / "val"
            ),
            "test": None,
            "nc": 2,
            "names": {
                0: "no_damage",
                1: "damage"
            },
            "channels": 3
        }

    def build_dataset(
        self,
        img_path,
        mode="train",
        batch=None
    ):

        return (
            IterativeClassificationDataset(
                image_dir=Path(img_path),
                label_dir=(
                    ITERATIVE_LABEL_ROOT
                    / mode
                ),
                args=self.args,
                augment=(
                    mode == "train"
                ),
                prefix=mode
            )
        )

    def get_validator(
        self
    ):

        return (
            IterativeClassificationValidator(
                self.test_loader,
                self.save_dir,
                args=copy(
                    self.args
                ),
                _callbacks=self.callbacks
            )
        )