from ultralytics import YOLO

from road_damage_detection.classifier.trainer import (
    IterativeClassificationTrainer
)

from road_damage_detection.config.paths import (
    ITERATIVE_ROOT,
    PROJECT_ROOT
)

from road_damage_detection.config.settings import (
    TRAIN_EPOCHS,
    TRAIN_IMAGE_SIZE,
    TRAIN_BATCH_SIZE,
    TRAIN_DEVICE,
    TRAIN_WORKERS,
    TRAIN_CLOSE_MOSAIC,
    TRAIN_DETERMINISTIC
)


def main():

    model = YOLO(
        "yolo11m-cls.pt"
    )

    model.train(
        data=str(
            ITERATIVE_ROOT
        ),
        trainer=(
            IterativeClassificationTrainer
        ),
        epochs=TRAIN_EPOCHS,
        imgsz=TRAIN_IMAGE_SIZE,
        batch=TRAIN_BATCH_SIZE,
        workers=TRAIN_WORKERS,
        device=TRAIN_DEVICE,
        deterministic = TRAIN_DETERMINISTIC,
        project=str(
            PROJECT_ROOT
            / "runs"
            / "training"
        ),
        name="iterative_yolo11m_classifier"
    )


if __name__ == "__main__":
    main()