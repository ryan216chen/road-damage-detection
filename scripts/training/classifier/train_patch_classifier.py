from pathlib import Path
from ultralytics import YOLO

FILE_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

DATA_ROOT = (
    FILE_ROOT
    / "data"
    / "histogram"
    / "classifier"
)

OUTPUT_ROOT = (
    FILE_ROOT
    / "runs"
    / "classification"
)

DATASET_NAMES = [
    "baseline",
    "equalized",
    "matched"
]

MODEL_NAME = "yolo11m-cls.pt"


def train(dataset_name):

    dataset_path = (
        DATA_ROOT
        / dataset_name
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset not found : "
            f"{dataset_path}"
        )

    print(
        f"[INFO] Dataset : "
        f"{dataset_path}"
    )

    model = YOLO(
        MODEL_NAME
    )

    model.train(
        data=str(dataset_path),
        epochs=50,
        imgsz=128,
        batch=32,
        workers=4,
        seed=42,
        deterministic=True,
        project=str(OUTPUT_ROOT),
        name=dataset_name
    )


def main():

    for dataset_name in DATASET_NAMES:

        train(
            dataset_name
        )


if __name__ == "__main__":
    main()