from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
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

ITERATIVE_LABEL_ROOT = (
    ITERATIVE_ROOT 
    / "labels"
)

#baseline

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


SEGMENTATION_IMAGE_ROOT = (
    SEGMENTATION_ROOT 
    / "images"
)

SEGMENTATION_LABEL_ROOT = (
    SEGMENTATION_ROOT 
    / "labels"
)


#equalization

HISTOGRAM_EQUALIZATION_ROOT = (
    PREPROCESSING_ROOT 
    / "histogram_equalization"
)

EQUALIZED_IMAGE_ROOT = (
    HISTOGRAM_EQUALIZATION_ROOT 
    / "images"
)

EQUALIZED_LABEL_ROOT = (
    HISTOGRAM_EQUALIZATION_ROOT 
    / "labels"
)




#matching


HISTOGRAM_MATCHING_ROOT = (
    PREPROCESSING_ROOT 
    / "histogram_matching"
)

HISTOGRAM_REFERENCE_ROOT = (
    HISTOGRAM_MATCHING_ROOT
    / "reference"
)

HISTOGRAM_REFERENCE_PATH = (
    HISTOGRAM_REFERENCE_ROOT
    / "road_l_hist.npy"
)

MATCHED_IMAGE_ROOT = (
    HISTOGRAM_MATCHING_ROOT
    / "images"
)

MATCHED_LABEL_ROOT = (
    HISTOGRAM_MATCHING_ROOT 
    / "labels"
)


#training 

TRAINING_ROOT = (
    PROJECT_ROOT 
    / "runs"
    / "training"
)

TRAINING_CCONFIG_ROOT = (
    TRAINING_ROOT 
    / "configs"
)