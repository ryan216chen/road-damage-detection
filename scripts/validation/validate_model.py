from pathlib import Path 
from ultralytics import YOLO 

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "road-segmentation"
    / "train_road_segmentation"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    PROJECT_ROOT 
    / "data"
    / "road_mask"
    / "data.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "validation"
)

def main():

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data = str(DATA_YAML),

        split = "val",

        imgsz = 640,
        batch = 16,
        workers = 4,
        device = 0,

        project = str(OUTPUT_DIR),
        name = "road_segmentation_val"
    )

    print()
    print("mAP50-95 :", metrics.box.map)
    print("mAP50 :", metrics.box.map50)
    print("precision :", metrics.box.mp)
    print("recall :", metrics.box.mr)

if __name__ == "__main__":
    main()