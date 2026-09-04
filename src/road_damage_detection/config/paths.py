from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
)

ITERATIVE_ROOT = (
    DATA_ROOT
    / "iterative"
)

ITERATIVE_IMAGE_ROOT = (
    ITERATIVE_ROOT
    / "images"
)

PREPROCESSING_ROOT = (
    DATA_ROOT
    / "preprocessing"
)

SEGMENTATION_ROOT = (
    PREPROCESSING_ROOT
    / "segmentation"
)

ROAD_MASK_ROOT = (
    SEGMENTATION_ROOT
    / "masks"
)