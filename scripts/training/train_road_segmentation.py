from ultralytics import YOLO 

DATA_YAML = r"D:\road-damage-detection\data\road_mask\data.yaml"

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

        project = r"D:\road-damage-detection\runs\road-segmentation",
        name = "train_road_segmentation"
    )

if __name__ == "__main__":
    main()