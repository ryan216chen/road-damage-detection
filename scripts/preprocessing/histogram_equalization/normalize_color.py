from pathlib import Path

import cv2
import numpy as np


# =========================
# Config
# =========================

INPUT_ROOT = Path(
    r"D:\rdd_project\data\yolo_dataset\images"
)

OUTPUT_ROOT = Path(
    r"D:\rdd_project\data\yolo_dataset_color_normalized\images"
)

TRAIN_DIR = (
    INPUT_ROOT
    / "train"
)

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png"
}


# =========================
# Calculate target statistics
# =========================

def calculate_train_statistics(
    train_dir: Path
):

    image_paths = [
        path
        for path in train_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_SUFFIXES
        )
    ]

    print(
        f"Train images : "
        f"{len(image_paths)}"
    )

    channel_sum = np.zeros(
        3,
        dtype=np.float64
    )

    channel_sq_sum = np.zeros(
        3,
        dtype=np.float64
    )

    pixel_count = 0

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        # Only resize for statistics
        # Makes calculation much faster
        image_small = cv2.resize(
            image,
            (256, 256)
        )

        lab = cv2.cvtColor(
            image_small,
            cv2.COLOR_BGR2LAB
        ).astype(
            np.float32
        )

        pixels = lab.reshape(
            -1,
            3
        )

        channel_sum += (
            pixels.sum(
                axis=0
            )
        )

        channel_sq_sum += (
            np.square(
                pixels
            ).sum(
                axis=0
            )
        )

        pixel_count += (
            pixels.shape[0]
        )

        if index % 500 == 0:

            print(
                f"Statistics : "
                f"{index}/"
                f"{len(image_paths)}"
            )

    mean = (
        channel_sum
        / pixel_count
    )

    variance = (
        channel_sq_sum
        / pixel_count
        - np.square(mean)
    )

    std = np.sqrt(
        variance
    )

    return (
        mean,
        std
    )


# =========================
# Normalize one image
# =========================

def normalize_image(
    image,
    target_mean,
    target_std
):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(
        np.float32
    )

    source_mean = lab.mean(
        axis=(0, 1)
    )

    source_std = lab.std(
        axis=(0, 1)
    )

    # Prevent division by zero
    source_std = np.maximum(
        source_std,
        1e-6
    )

    normalized = (
        (
            lab
            - source_mean
        )
        / source_std
    )

    normalized = (
        normalized
        * target_std
        + target_mean
    )

    normalized = np.clip(
        normalized,
        0,
        255
    ).astype(
        np.uint8
    )

    result = cv2.cvtColor(
        normalized,
        cv2.COLOR_LAB2BGR
    )

    return result


# =========================
# Process dataset
# =========================

def process_dataset(
    target_mean,
    target_std
):

    image_paths = [
        path
        for path in INPUT_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_SUFFIXES
        )
    ]

    print(
        f"Total images : "
        f"{len(image_paths)}"
    )

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Failed : "
                f"{image_path}"
            )

            continue

        normalized = normalize_image(
            image=image,
            target_mean=target_mean,
            target_std=target_std
        )

        relative_path = (
            image_path.relative_to(
                INPUT_ROOT
            )
        )

        output_path = (
            OUTPUT_ROOT
            / relative_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(output_path),
            normalized
        )

        if index % 100 == 0:

            print(
                f"[{index}/"
                f"{len(image_paths)}] "
                f"{relative_path}"
            )


def main():

    print(
        "Calculating training "
        "color statistics..."
    )

    (
        target_mean,
        target_std
    ) = calculate_train_statistics(
        TRAIN_DIR
    )

    print(
        "Target mean : "
        f"{target_mean}"
    )

    print(
        "Target std : "
        f"{target_std}"
    )

    print(
        "\nApplying color "
        "normalization..."
    )

    process_dataset(
        target_mean=target_mean,
        target_std=target_std
    )


if __name__ == "__main__":
    main()