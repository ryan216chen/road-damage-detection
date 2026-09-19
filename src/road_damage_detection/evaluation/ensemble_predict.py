from ensemble_boxes import weighted_boxes_fusion 
from tqdm import tqdm 
from ultralytics import YOLO 
from road_damage_detection.config.paths import (
    PROJECT_ROOT,
    TEST_IMAGE_ROOT
)

from road_damage_detection.config.settings import (
    TRAIN_DEVICE,
    TRAIN_IMAGE_SIZE
)
import shutil 

WEIGHT_PATHS = [
    (
        PROJECT_ROOT
        / "runs"
        / "training"
        / "iterative_yolo26m"
        / "weights"
        / "best.pt"
    ),
    (
        PROJECT_ROOT
        / "runs"
        / "training"
        / "iterative_yolo11m"
        / "weights"
        / "best.pt"
    )
]

MODEL_WEIGHTS = [
    1.0,
    1.0
]

BATCH_SIZE = 64 
CONFIDENCE = 0.25 
IOU_THRESHOLD = 0.55 

def get_predictions(
    result 
):

    height, width = result.orig_shape 

    boxes = []
    scores = []
    labels = []

    for box in result.boxes:
        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .tolist()
        )

        boxes.append([
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height 
        ])

        scores.append(
            float(
                box.conf.item()
            )
        )

        labels.append(
            int(
                box.cls.item()
            )
        )

    return (
        boxes,
        scores,
        labels 
    )

def fuse_results(
    results 
):

    boxes_list = []
    scores_list = []
    labels_list = []

    height, width = (
        results[0]
        .orig_shape
    )

    for result in results:

        boxes, scores, labels = (
            get_predictions(
                result 
            )
        )

        boxes_list.append(boxes)

        scores_list.append(scores)

        labels_list.append(labels)

    boxes, scores, labels = (
        weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights = MODEL_WEIGHTS,
            iou_thr = IOU_THRESHOLD,
            skip_box_thr = CONFIDENCE
        )
    )

    return [
        (
            int(label),
            float(score),
            [
                box[0] * width,
                box[1] * height,
                box[2] * width,
                box[3] * height 
            ]
        )
        for box, score, label 
        in zip(
            boxes,
            scores,
            labels 
        )
    ]

def main():

    models = [
        YOLO(
            str(path)
        )
        for path in WEIGHT_PATHS
    ]

    image_paths = sorted(
        TEST_IMAGE_ROOT.rglob("*.jpg")
    )

    output_dir = (
        PROJECT_ROOT
        / "runs"
        / "prediction"
        / "ensemble_yolo26m_yolo11m"
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = (
        output_dir 
        / "submission.csv"
    )

    with open(
        submission_path,
        "w",
        encoding="utf-8"
    ) as file:

        for start in tqdm(
            range(
                0,
                len(image_paths),
                BATCH_SIZE
            ),
            desc = "Ensemble",
            unit = "batch"
        ):

            batch_paths = image_paths[
                start:start + BATCH_SIZE
            ]

            batch_sources = [
                str(path)
                for path in batch_paths 
            ]

            model_results = [
                model.predict(
                    source = batch_sources,
                    imgsz = TRAIN_IMAGE_SIZE,
                    conf = CONFIDENCE,
                    device = TRAIN_DEVICE,
                    verbose = False 
                )
                for model in models 
            ]

            for index, image_path in enumerate(
                batch_paths
            ):

                results = [
                    model_result[
                        index 
                    ]
                    for model_result in model_results 
                ]

                predictions = fuse_results(
                    results 
                )

                output = []

                for (
                    class_id,
                    confidence,
                    box 
                ) in predictions:

                    x1, y1, x2, y2 = box 

                    output.extend([
                        str(
                            class_id + 1
                        ),
                        str(
                            round(x1)
                        ),
                        str(
                            round(y1)
                        ),
                        str(
                            round(x2)
                        ),
                        str(
                            round(y2)
                        )
                    ])

                file.write(
                    f"{image_path.name},"
                    f"{' '.join(output)}\n"
                )

    
    print(
        f"Saved to : "
        f"{submission_path}"
    )

if __name__ == "__main__":
    main()




