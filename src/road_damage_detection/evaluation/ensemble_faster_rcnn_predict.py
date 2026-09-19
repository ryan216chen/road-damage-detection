import shutil

import torch

from ensemble_boxes import (
    weighted_boxes_fusion
)

from PIL import Image

from tqdm import tqdm

from torchvision.transforms.functional import (
    pil_to_tensor
)

from ultralytics import YOLO

from road_damage_detection.config.faster_rcnn.runs import (
    get_checkpoint
)

from road_damage_detection.config.faster_rcnn.settings import (
    FASTER_RCNN_BATCH_SIZE
)

from road_damage_detection.config.paths import (
    PROJECT_ROOT,
    TEST_IMAGE_ROOT
)

from road_damage_detection.config.settings import (
    TRAIN_DEVICE,
    TRAIN_IMAGE_SIZE
)

from road_damage_detection.models.faster_rcnn.model import (
    build_model
)


YOLO_WEIGHT_PATH = (
    PROJECT_ROOT
    / "runs"
    / "training"
    / "iterative_yolo26m"
    / "weights"
    / "best.pt"
)

FASTER_RCNN_WEIGHT_PATH = (
    get_checkpoint(
        "iterative"
    )
)

MODEL_WEIGHTS = [
    1.0,
    0.5
]

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.55


def get_yolo_predictions(
    result
):

    height, width = (
        result.orig_shape
    )

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


def get_faster_rcnn_predictions(
    output,
    width,
    height
):

    boxes = []
    scores = []
    labels = []

    for (
        box,
        score,
        label
    ) in zip(
        output["boxes"],
        output["scores"],
        output["labels"]
    ):

        score = float(
            score.item()
        )

        if score < CONFIDENCE:
            continue

        x1, y1, x2, y2 = (
            box
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
            score
        )

        labels.append(
            int(
                label.item()
            )
            - 1
        )

    return (
        boxes,
        scores,
        labels
    )


def load_faster_rcnn_images(
    image_paths,
    device
):

    images = []

    for image_path in image_paths:

        with Image.open(
            image_path
        ) as image:

            image = (
                image
                .convert(
                    "RGB"
                )
            )

            image = (
                pil_to_tensor(
                    image
                )
                .float()
                / 255.0
            )

        images.append(
            image.to(
                device
            )
        )

    return images


def fuse_results(
    yolo_result,
    faster_rcnn_output
):

    height, width = (
        yolo_result.orig_shape
    )

    (
        yolo_boxes,
        yolo_scores,
        yolo_labels
    ) = get_yolo_predictions(
        yolo_result
    )

    (
        faster_rcnn_boxes,
        faster_rcnn_scores,
        faster_rcnn_labels
    ) = get_faster_rcnn_predictions(
        faster_rcnn_output,
        width,
        height
    )

    boxes, scores, labels = (
        weighted_boxes_fusion(
            [
                yolo_boxes,
                faster_rcnn_boxes
            ],
            [
                yolo_scores,
                faster_rcnn_scores
            ],
            [
                yolo_labels,
                faster_rcnn_labels
            ],
            weights=MODEL_WEIGHTS,
            iou_thr=IOU_THRESHOLD,
            skip_box_thr=CONFIDENCE
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

    device = torch.device(
        f"cuda:{TRAIN_DEVICE}"
        if torch.cuda.is_available()
        else "cpu"
    )

    yolo_model = YOLO(
        str(
            YOLO_WEIGHT_PATH
        )
    )

    faster_rcnn_model = (
        build_model()
    )

    checkpoint = torch.load(
        FASTER_RCNN_WEIGHT_PATH,
        map_location=device,
        weights_only=False
    )

    faster_rcnn_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    faster_rcnn_model.to(
        device
    )

    faster_rcnn_model.eval()

    image_paths = sorted(
        TEST_IMAGE_ROOT.rglob(
            "*.jpg"
        )
    )

    output_dir = (
        PROJECT_ROOT
        / "runs"
        / "prediction"
        / "ensemble_yolo26m_faster_rcnn"
    )

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

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
                FASTER_RCNN_BATCH_SIZE
            ),
            desc="YOLO26m + Faster R-CNN",
            unit="batch"
        ):

            batch_paths = image_paths[
                start:
                start
                + FASTER_RCNN_BATCH_SIZE
            ]

            batch_sources = [
                str(path)
                for path in batch_paths
            ]

            yolo_results = (
                yolo_model.predict(
                    source=batch_sources,
                    imgsz=TRAIN_IMAGE_SIZE,
                    conf=CONFIDENCE,
                    device=TRAIN_DEVICE,
                    verbose=False
                )
            )

            faster_rcnn_images = (
                load_faster_rcnn_images(
                    batch_paths,
                    device
                )
            )

            with torch.inference_mode():

                faster_rcnn_results = (
                    faster_rcnn_model(
                        faster_rcnn_images
                    )
                )

            for (
                image_path,
                yolo_result,
                faster_rcnn_result
            ) in zip(
                batch_paths,
                yolo_results,
                faster_rcnn_results
            ):

                predictions = (
                    fuse_results(
                        yolo_result,
                        faster_rcnn_result
                    )
                )

                output = []

                for (
                    class_id,
                    confidence,
                    box
                ) in predictions:

                    x1, y1, x2, y2 = (
                        box
                    )

                    output.extend([
                        str(
                            class_id
                            + 1
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
