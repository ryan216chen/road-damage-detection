from pathlib import Path
import shutil

import cv2
import numpy as np


# =========================
# Config
# =========================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[4]
)

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "iterative"
)

INPUT_IMAGE_ROOT = (
    INPUT_ROOT
    / "images"
)

INPUT_LABEL_ROOT = (
    INPUT_ROOT
    / "labels"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "l_histogram_matching"
)

OUTPUT_IMAGE_ROOT = (
    OUTPUT_ROOT
    / "images"
)

OUTPUT_LABEL_ROOT = (
    OUTPUT_ROOT
    / "labels"
)

TRAIN_DIR = (
    INPUT_IMAGE_ROOT
    / "train"
)

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png"
}


# =========================
# Get image paths
# =========================

def get_image_paths(
    root: Path
):

    return [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_SUFFIXES
        )
    ]


# =========================
# Calculate target L histogram
# =========================

def calculate_target_histogram(
    train_dir: Path
):

    image_paths = get_image_paths(
        train_dir
    )

    print(
        f"Train images : "
        f"{len(image_paths)}"
    )

    target_hist = np.zeros(
        256,
        dtype=np.float64
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

        # Resize only for calculating
        # target histogram.
        #
        # We do NOT save this resized image.
        image_small = cv2.resize(
            image,
            (256, 256)
        )

        lab = cv2.cvtColor(
            image_small,
            cv2.COLOR_BGR2LAB
        )

        l_channel = lab[:, :, 0]

        hist = np.bincount(
            l_channel.ravel(),
            minlength=256
        )

        target_hist += hist

        if index % 500 == 0:

            print(
                f"Target histogram : "
                f"{index}/"
                f"{len(image_paths)}"
            )

    total_pixels = target_hist.sum()

    if total_pixels == 0:

        raise RuntimeError(
            "No valid pixels found "
            "in training images."
        )

    target_hist /= total_pixels

    return target_hist


# =========================
# Convert histogram to CDF
# =========================

def histogram_to_cdf(
    histogram
):

    cdf = np.cumsum(
        histogram
    )

    cdf /= cdf[-1]

    return cdf


# =========================
# Match one L channel
# =========================

def match_l_channel(
    l_channel,
    target_cdf
):

    source_hist = np.bincount(
        l_channel.ravel(),
        minlength=256
    ).astype(
        np.float64
    )

    source_hist /= (
        source_hist.sum()
    )

    source_cdf = histogram_to_cdf(
        source_hist
    )

    # target_cdf may contain repeated
    # values when some intensity bins
    # have zero pixels.
    #
    # np.interp works better if the
    # x values are unique.
    (
        target_cdf_unique,
        unique_indices
    ) = np.unique(
        target_cdf,
        return_index=True
    )

    target_values = np.arange(
        256
    )[unique_indices]

    lookup_table = np.interp(
        source_cdf,
        target_cdf_unique,
        target_values
    )

    lookup_table = np.clip(
        lookup_table,
        0,
        255
    ).astype(
        np.uint8
    )

    matched_l = lookup_table[
        l_channel
    ]

    return matched_l


# =========================
# Process one image
# =========================

def match_image(
    image,
    target_cdf
):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    (
        l_channel,
        a_channel,
        b_channel
    ) = cv2.split(
        lab
    )

    matched_l = match_l_channel(
        l_channel=l_channel,
        target_cdf=target_cdf
    )

    matched_lab = cv2.merge([
        matched_l,
        a_channel,
        b_channel
    ])

    result = cv2.cvtColor(
        matched_lab,
        cv2.COLOR_LAB2BGR
    )

    return result


# =========================
# Process dataset
# =========================

def process_dataset(
    target_cdf
):

    image_paths = get_image_paths(
        INPUT_IMAGE_ROOT
    )

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

        matched = match_image(
            image=image,
            target_cdf=target_cdf
        )

        relative_path = (
            image_path.relative_to(
                INPUT_IMAGE_ROOT
            )
        )

        output_path = (
            OUTPUT_IMAGE_ROOT
            / relative_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        success = cv2.imwrite(
            str(output_path),
            matched
        )

        if not success:

            print(
                f"Write failed : "
                f"{output_path}"
            )

        if index % 100 == 0:

            print(
                f"[{index}/"
                f"{len(image_paths)}] "
                f"{relative_path}"
            )


# =========================
# Copy labels
# =========================

def copy_labels():

    if not INPUT_LABEL_ROOT.exists():

        print(
            "Label directory "
            "does not exist."
        )

        return

    shutil.copytree(
        INPUT_LABEL_ROOT,
        OUTPUT_LABEL_ROOT,
        dirs_exist_ok=True
    )

    print(
        "Labels copied."
    )


# =========================
# Main
# =========================

def main():

    print(
        "Calculating target "
        "L-channel histogram..."
    )

    target_hist = (
        calculate_target_histogram(
            TRAIN_DIR
        )
    )

    target_cdf = (
        histogram_to_cdf(
            target_hist
        )
    )

    print(
        "Applying L-channel "
        "histogram matching..."
    )

    process_dataset(
        target_cdf=target_cdf
    )

    print(
        "Copying labels..."
    )

    copy_labels()

    print(
        "Done."
    )

    print(
        f"Output : "
        f"{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()