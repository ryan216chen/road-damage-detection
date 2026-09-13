from copy import copy 

from ultralytics.models.yolo.detect import (
    DetectionTrainer,
    DetectionValidator
)

RGB_MEAN = (
    0.5045821027,
    0.5066387759,
    0.4853259145
)

def substract_mean(
    image 
):

    mean = image.new_tensor(
        RGB_MEAN 
    ).view(
        1,
        3,
        1,
        1
    )

    return (
        image - mean 
    )

class MeanSubtractionValidator(
    DetectionValidator
):

    def preprocess(
        self,
        batch
    ):

        batch = super().preprocess(batch)

        batch["img"] = substract_mean(batch["img"])

        return batch 

class MeanSubtractionTrainer(
    DetectionTrainer
):

    def preprocess_batch(
        self,
        batch 
    ):

        batch = super().preprocess_batch(batch)

        batch["img"] = substract_mean(batch["img"])

        return batch 

    def get_validator(
        self
    ):

        return MeanSubtractionValidator(
            self.test_loader,
            save_dir = self.save_dir,
            args = copy(
                self.args
            ),
            _callbacks = self.callbacks 
        )