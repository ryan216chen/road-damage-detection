from ultralytics import YOLO 
from pathlib import Path 

LAST_MODEL = Path(r)

def main():

    model = YOLO(LAST_MODEL)

    model.train(
        resume=True
    )

if __name__ == "__main__":
    main()