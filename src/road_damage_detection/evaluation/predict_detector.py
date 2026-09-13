import argparse
import shutil
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

from road_damage_detection.config.paths import (
    PROJECT_ROOT,
    TEST_IMAGE_ROOT
)

from road_damage_detection.config.settings import (
    TRAIN_IMAGE_SIZE,
    TRAIN_DEVICE,
    YOLO_MODEL
)

from road_damage_detection.config.datasets import (
    add_dataset_argument
)


TRAINING_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "training"
)

PREDICTION_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "prediction"
)

BATCH_SIZE = 64


def predict(
    dataset_name,
    source=None,
    conf=0.25
):

    if source is None:
        source_path = TEST_IMAGE_ROOT
    else:
        source_path = Path(source)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source not found : {source_path}"
        )


    image_paths = [
        str(path)
        for path in source_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".jpg"
    ]


    model_path = (
        TRAINING_ROOT
        / f"{dataset_name}_{Path(YOLO_MODEL).stem}"
        / "weights"
        / "best.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found : {model_path}"
        )


    print(f"[INFO] Dataset : {dataset_name}")

    print(f"[INFO] Model : {model_path}")

    print(f"[INFO] Source : {source_path}")


    output_path = (
        PREDICTION_ROOT
        / dataset_name
    )

    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )


    model = YOLO(model_path)


    submission_path = (
        PREDICTION_ROOT
        / dataset_name
        / "submission.csv"
    )


    with open(
        submission_path,
        "w",
        encoding="utf-8"
    ) as file:

        with tqdm(
            total=len(image_paths),
            desc="Predicting",
            unit="image"
        ) as progress_bar:

            for start in range(
                0,
                len(image_paths),
                BATCH_SIZE
            ):

                batch_paths = image_paths[
                    start:start + BATCH_SIZE
                ]


                results = model.predict(
                    source=batch_paths,
                    imgsz=TRAIN_IMAGE_SIZE,
                    conf=conf,
                    device=TRAIN_DEVICE,
                    save=True,
                    save_txt=True,
                    project=str(PREDICTION_ROOT),
                    name=dataset_name,
                    exist_ok=True,
                    stream=True,
                    verbose=False
                )


                for result in results:

                    image_name = Path(
                        result.path
                    ).name


                    predictions = []

                    if result.boxes is not None:

                        for box in result.boxes:

                            class_id = (
                                int(box.cls.item())
                                + 1
                            )

                            x1, y1, x2, y2 = (
                                box.xyxy[0]
                                .cpu()
                                .tolist()
                            )


                            predictions.extend([
                                str(class_id),
                                str(int(round(x1))),
                                str(int(round(y1))),
                                str(int(round(x2))),
                                str(int(round(y2)))
                            ])


                    prediction_string = " ".join(
                        predictions
                    )

                    file.write(
                        f"{image_name},{prediction_string}\n"
                    )


                    progress_bar.update(1)


    images_output_path = (
        output_path
        / "images"
    )

    images_output_path.mkdir(
        parents=True,
        exist_ok=True
    )


    for path in output_path.iterdir():

        if (
            path.is_file()
            and path.suffix.lower() == ".jpg"
        ):

            shutil.move(
                str(path),
                str(
                    images_output_path
                    / path.name
                )
            )


    print(
        f"[INFO] Submission saved to : "
        f"{submission_path}"
    )

    print(
        f"[INFO] Confidence : {conf}"
    )


def main():

    parser = argparse.ArgumentParser()

    add_dataset_argument(parser)

    parser.add_argument(
        "--source",
        type=str,
        default=None
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    args = parser.parse_args()

    predict(
        dataset_name=args.dataset,
        source=args.source,
        conf=args.conf
    )


if __name__ == "__main__":
    main()