from ultralytics import YOLO 
from pathlib import Path 

LAST_MODEL = Path(r"D:\road-damage-detection\runs\road-segmentation\train_road_segmentation\weights\last.pt")

def main():

    model = YOLO(LAST_MODEL)

    model.train(
        resume=True
    )

if __name__ == "__main__":
    main()