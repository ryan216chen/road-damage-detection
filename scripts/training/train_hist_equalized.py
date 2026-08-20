from ultralytics import YOLO 

DATA_YAML = 

MODEL = "yolo11m.pt"

def main():

    model = YOLO(
        MODEL 
    )

    model.train(
        data = DATA_YAML,
        epochs = 150,
        imgsz = 640,
        batch = 16,
        workers = 4,
        device = 0,

        project =  ,
        name = 
    )

if __name__ == "__main__":
    main()