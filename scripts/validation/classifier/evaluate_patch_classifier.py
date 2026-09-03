from pathlib import Path 
from ultralytics import YOLO 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

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

RUN_ROOT = (
    FILE_ROOT 
    / "runs"
    / "classification"
)

def evaluate(dataset_name):


    model_path = (
        RUN_ROOT 
        / dataset_name 
        / "weights"
        / "best.pt"
    )

    test_root = (
        DATA_ROOT 
        / dataset_name 
        / "test"
    )

    model = YOLO(str(model_path))

    y_true = []
    y_pred = []

    for label_dir in test_root.iterdir():

        if not label_dir.is_dir():
            continue 

        true_label = label_dir.name 

        for image_path in label_dir.rglob("*.jpg"):

            result = model.predict(
                str(image_path),
                verbose = False
            )[0]


            pred_id = result.probs.top1 

            pred_label = result.names[pred_id]

            y_true.append(true_label)

            y_pred.append(pred_label)

    print(f"\n===={dataset_name}====")


    print(
        "Accuracy : ",
        accuracy_score(y_true, y_pred)
    )

    print(
        classification_report(
            y_true,
            y_pred,
            digits=4
        )
    )

    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=["crack", "normal"]
        )
    )

def main():

    evaluate("baseline")
    evaluate("equalized")
    evaluate("matched")

if __name__ == "__main__":
    main()