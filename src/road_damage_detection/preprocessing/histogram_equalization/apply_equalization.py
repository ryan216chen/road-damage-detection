import shutil

from concurrent.futures import (
    ProcessPoolExecutor
)

import cv2
import numpy as np

from tqdm import tqdm

from road_damage_detection.config.paths import (
    ITERATIVE_IMAGE_ROOT,
    ITERATIVE_LABEL_ROOT,
    ROAD_MASK_ROOT,
    EQUALIZED_IMAGE_ROOT,
    EQUALIZED_LABEL_ROOT
)

from road_damage_detection.config.settings import (
    IMAGE_EXTENSIONS,
    EQUALIZATION_WORKERS
)


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

    cv2.setNumThreads(1)


def equalize_road(
    image,
    mask
):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

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

    road_l = (
        l[road_mask]
    )

    equalized_l = (
        cv2.equalizeHist(
            road_l.reshape(
                -1,
                1
            )
        )
        .reshape(-1)
    )

    new_l = (
        l.copy()
    )

    new_l[
        road_mask
    ] = equalized_l

    equalized_lab = (
        cv2.merge(
            [
                new_l,
                a,
                b
            ]
        )
    )

    result = cv2.cvtColor(
        equalized_lab,
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
    ) = equalize_road(
        image,
        mask
    )

    output_image_path = (
        EQUALIZED_IMAGE_ROOT
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
        EQUALIZED_LABEL_ROOT
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
        EQUALIZED_IMAGE_ROOT
    )

    clear_output_root(
        EQUALIZED_LABEL_ROOT
    )

    EQUALIZED_IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    EQUALIZED_LABEL_ROOT.mkdir(
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
        f"{EQUALIZATION_WORKERS}"
    )

    if not image_paths:

        raise FileNotFoundError(
            f"No images found in : "
            f"{ITERATIVE_IMAGE_ROOT}"
        )

    empty_masks = 0

    with ProcessPoolExecutor(
        max_workers=EQUALIZATION_WORKERS,
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
            desc="Histogram equalization"
        ):

            empty_masks += (
                empty_mask
            )

    print(
        f"[INFO] Empty masks : "
        f"{empty_masks}"
    )


def main():

    process_dataset()

    print(
        "[INFO] Histogram equalization completed."
    )


if __name__ == "__main__":

    main()