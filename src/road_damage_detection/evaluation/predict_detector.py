import argparse
import shutil
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

from road_damage_detection.config.datasets import (
    add_dataset_argument
)
from road_damage_detection.config.experiments import (
    add_experiment_arguments,
    resolve_experiment
)
from road_damage_detection.config.paths import (
    PROJECT_ROOT,
    TEST_IMAGE_ROOT
)
from road_damage_detection.config.runs import (
    PREDICTION_ROOT,
    get_checkpoint,
    get_run_dir_from_checkpoint,
    get_run_name
)
from road_damage_detection.config.settings import (
    TRAIN_IMAGE_SIZE,
    TRAIN_DEVICE,
)


BATCH_SIZE = 64


def predict(
    dataset_name,
    source=None,
    conf=0.25,
    weights=None,
    experiment_name=None,
    mean_subtraction=False
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

    experiment = resolve_experiment(
        experiment_name,
        mean_subtraction
    )

    run_name = get_run_name(
        dataset_name,
        experiment
    )

    if weights is None:

        model_path = get_checkpoint(
            run_name,
            filename="best.pt"
        )

    else:

        model_path = Path(
            weights
        )

        if not model_path.is_absolute():
            model_path = (
                PROJECT_ROOT
                / model_path
            )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found : "
                f"{model_path}"
            )

    actual_run_name = (
        get_run_dir_from_checkpoint(
            model_path
        ).name
    )

    print(
        f"[INFO] Dataset : "
        f"{dataset_name}"
    )

    print(
        f"[INFO] Experiment : "
        f"{experiment.name}"
    )

    print(
        f"[INFO] Model : "
        f"{model_path}"
    )

    print(
        f"[INFO] Source : "
        f"{source_path}"
    )

    output_path = (
        PREDICTION_ROOT
        / actual_run_name
    )

    if output_path.exists():
        shutil.rmtree(
            output_path
        )

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    model = YOLO(
        model_path
    )

    submission_path = (
        output_path
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

                predict_args = {
                    "source": batch_paths,
                    "imgsz": TRAIN_IMAGE_SIZE,
                    "conf": conf,
                    "device": TRAIN_DEVICE,
                    "save": True,
                    "save_txt": True,
                    "project": str(PREDICTION_ROOT),
                    "name": actual_run_name,
                    "exist_ok": True,
                    "stream": True,
                    "verbose": False
                }

                if experiment.predictor is not None:
                    predict_args["predictor"] = (
                        experiment.predictor
                    )

                results = model.predict(
                    **predict_args
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

    add_dataset_argument(
        parser
    )

    add_experiment_arguments(
        parser
    )

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

    parser.add_argument(
        "--weights",
        default=None
    )

    args = parser.parse_args()

    predict(
        dataset_name=args.dataset,
        source=args.source,
        conf=args.conf,
        weights=args.weights,
        experiment_name=args.experiment,
        mean_subtraction=args.mean_subtraction
    )


if __name__ == "__main__":
    main()
