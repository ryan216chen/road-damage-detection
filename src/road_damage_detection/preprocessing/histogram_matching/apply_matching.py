import shutil

from concurrent.futures import (
    ProcessPoolExecutor
)

import cv2
import numpy as np

from skimage.exposure import (
    match_histograms
)

from tqdm import tqdm

from road_damage_detection.config.paths import (
    ITERATIVE_IMAGE_ROOT,
    ITERATIVE_LABEL_ROOT,
    ROAD_MASK_ROOT,
    HISTOGRAM_REFERENCE_PATH,
    MATCHED_IMAGE_ROOT,
    MATCHED_LABEL_ROOT
)

from road_damage_detection.config.settings import (
    IMAGE_EXTENSIONS,
    MATCHING_WORKERS
)


REFERENCE_L = None


def get_image_paths():

    image_paths = [
        path
        for path
        in ITERATIVE_IMAGE_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    return image_paths


def clear_output_root(
    root
):

    if root.exists():

        print(
            f"[INFO] Removing existing output : "
            f"{root}"
        )

        if root.is_dir():

            shutil.rmtree(
                root
            )

        else:

            root.unlink()


def init_worker():

    global REFERENCE_L

    cv2.setNumThreads(1)

    REFERENCE_L = np.load(
        HISTOGRAM_REFERENCE_PATH,
        mmap_mode="r"
    )


def apply_matching(
    image,
    mask
):

    road_mask = (
        mask > 0
    )

    if not np.any(
        road_mask
    ):

        return (
            np.zeros_like(
                image
            ),
            True
        )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    road_l = (
        l[road_mask]
    )

    matched_l = (
        match_histograms(
            road_l,
            REFERENCE_L
        )
    )

    matched_l = (
        matched_l
        .clip(
            0,
            255
        )
        .astype(
            np.uint8
        )
    )

    new_l = (
        l.copy()
    )

    new_l[
        road_mask
    ] = matched_l

    matched_lab = (
        cv2.merge(
            [
                new_l,
                a,
                b
            ]
        )
    )

    result = cv2.cvtColor(
        matched_lab,
        cv2.COLOR_LAB2BGR
    )

    result[
        ~road_mask
    ] = 0

    return (
        result,
        False
    )


def process_image(
    image_path
):

    relative_path = (
        image_path
        .relative_to(
            ITERATIVE_IMAGE_ROOT
        )
    )

    mask_path = (
        ROAD_MASK_ROOT
        / relative_path
    ).with_suffix(
        ".png"
    )

    if not mask_path.exists():

        raise FileNotFoundError(
            f"Mask not found : "
            f"{mask_path}"
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Failed to read image : "
            f"{image_path}"
        )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        raise RuntimeError(
            f"Failed to read mask : "
            f"{mask_path}"
        )

    (
        result,
        empty_mask
    ) = apply_matching(
        image,
        mask
    )

    output_image_path = (
        MATCHED_IMAGE_ROOT
        / relative_path
    )

    output_image_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(output_image_path),
        result
    )

    if not success:

        raise RuntimeError(
            f"Failed to write image : "
            f"{output_image_path}"
        )

    label_relative_path = (
        relative_path
        .with_suffix(
            ".txt"
        )
    )

    source_label_path = (
        ITERATIVE_LABEL_ROOT
        / label_relative_path
    )

    output_label_path = (
        MATCHED_LABEL_ROOT
        / label_relative_path
    )

    if not source_label_path.exists():

        raise FileNotFoundError(
            f"Label not found : "
            f"{source_label_path}"
        )

    output_label_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        source_label_path,
        output_label_path
    )

    return int(
        empty_mask
    )


def process_dataset():

    clear_output_root(
        MATCHED_IMAGE_ROOT
    )

    clear_output_root(
        MATCHED_LABEL_ROOT
    )

    MATCHED_IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    MATCHED_LABEL_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = (
        get_image_paths()
    )

    print(
        f"[INFO] Images : "
        f"{len(image_paths)}"
    )

    print(
        f"[INFO] Workers : "
        f"{MATCHING_WORKERS}"
    )

    if not image_paths:

        raise FileNotFoundError(
            f"No images found in : "
            f"{ITERATIVE_IMAGE_ROOT}"
        )

    empty_masks = 0

    with ProcessPoolExecutor(
        max_workers=MATCHING_WORKERS,
        initializer=init_worker
    ) as executor:

        results = executor.map(
            process_image,
            image_paths,
            chunksize=8
        )

        for empty_mask in tqdm(
            results,
            total=len(image_paths),
            desc="Histogram matching"
        ):

            empty_masks += (
                empty_mask
            )

    print(
        f"[INFO] Empty masks : "
        f"{empty_masks}"
    )


def main():

    if not HISTOGRAM_REFERENCE_PATH.exists():

        raise FileNotFoundError(
            f"Reference not found : "
            f"{HISTOGRAM_REFERENCE_PATH}"
        )

    reference_l = np.load(
        HISTOGRAM_REFERENCE_PATH,
        mmap_mode="r"
    )

    print(
        f"[INFO] Reference pixels : "
        f"{len(reference_l)}"
    )

    print(
        f"[INFO] Loading reference : "
        f"{HISTOGRAM_REFERENCE_PATH}"
    )

    process_dataset()

    print(
        "[INFO] Histogram matching completed."
    )


if __name__ == "__main__":

    main()