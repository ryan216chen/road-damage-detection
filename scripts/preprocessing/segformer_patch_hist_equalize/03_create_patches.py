from pathlib import Path 
import csv 
import cv2 
import shutil 

FILE_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

RAW_IMAGE_ROOT = (
    FILE_ROOT 
    / "data"
    / "segformer_road"
    / "images"
    / "train"
)

EQUALIZED_IMAGE_ROOT = (
    FILE_ROOT 
    / "data"
    / "histogram_equalized"
    / "images"
    / "train"
)

MASK_ROOT = (
    FILE_ROOT
    / "data"
    / "segformer_road"
    / "masks"
    / "train"
)

OUTPUT_ROOT = (
    FILE_ROOT
    / "data"
    / "patch_dataset"
)

RAW_OUTPUT_ROOT = (
    OUTPUT_ROOT 
    / "raw"
)

EQUALIZED_OUTPUT_ROOT = (
    OUTPUT_ROOT 
    / "equalized"
)

CSV_PATH = (
    OUTPUT_ROOT 
    / "patches.csv"
)

GRID = 4
MIN_ROAD_RATIO = 0.7


def get_patch(
    image,
    row,
    col 
):

    height, width = image.shape[:2]

    patch_height = height // GRID 

    patch_width = width // GRID 

    y1 = row * patch_height 

    y2 = (
        height 
        if row == GRID - 1
        else (row + 1) * patch_height 
    )

    x1 = col * patch_width 

    x2 = (
        width
        if col == GRID - 1
        else (col + 1) * patch_width 
    )

    patch = (
        image[
            y1:y2,
            x1:x2 
        ]
    )

    return (
        patch,
        x1,
        y1,
        x2,
        y2 
    )

def get_candidates(
    mask 
):

    candidates = []

    for row in range(GRID):

        for col in range(GRID):

            (
                patch,
                _,
                _,
                _,
                _ 
            ) = get_patch(
                mask,
                row,
                col 
            )

            road_ratio = (
                patch > 0
            ).mean()

            if (
                road_ratio >= MIN_ROAD_RATIO
            ):

                candidates.append(
                    (
                        row,
                        col,
                        road_ratio 
                    )
                )

    return candidates 