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
    / "classifier_dataset"
)

OUTPUT_ROOT = (
    FILE_ROOT 
    / "runs"
    / "classification"
)

DATASET_NAME = "equalized"

MODEL_NAME = "yolo11m-cls.pt"
 

def train():

    dataset_path = (
        DATA_ROOT 
        / DATASET_NAME 
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

    model = YOLO(MODEL_NAME)

    model.train(
        data = str(dataset_path),
        epochs = 50,
        imgsz = 128,
        batch = 32,
        workers = 4,
        seed = 42,
        deterministic = True,
        project = str(OUTPUT_ROOT),
        name = DATASET_NAME 
    )

def main():

    train()

if __name__ == "__main__":
    main()