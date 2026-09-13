from copy import copy 

from ultralytics.models.yolo.detect import (
    DetectionTrainer,
    DetectionValidator,
    DetectionPredictor
)

RGB_MEAN = (
    0.5045821027,
    0.5066387759,
    0.4853259145
)

RGB_STD = (
    0.2839588386,
    0.2891190913,
    0.3154811514

)

def standardize(
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

    std = image.new_tensor(
        RGB_STD
    ).view(
        1,
        3,
        1,
        1
    )

    return (
        (image - mean)
        / std 
    )


class StandardizationValidator(
    DetectionValidator 
):

    def preprocess(
        self,
        batch 
    ):

        batch = super().preprocess(batch)

        batch["img"] = standardize(batch["img"])

        return batch 

class StandardizationTrainer(
    DetectionTrainer 
):

    def preprocess_batch(
        self,
        batch 
    ):

        batch = super().preprocess_batch(batch)

        batch["img"] = standardize(batch["img"])

        return batch 

    
    def get_validator(
        self 
    ):

        return StandardizationValidator(
            self.test_loader,
            save_dir = self.save_dir,
            args = copy(self.args),
            _callbacks = self.callbacks 
        )

class StandardizationPredictor(
    DetectionPredictor
):

    def preprocess(
        self,
        image
    ):

        image = super().preprocess(
            image
        )

        image = standardize(
            image
        )

        return image