import cv2 
import numpy as np 
from tqdm import tqdm 

from road_damage_detection.config.paths import (
    ITERATIVE_IMAGE_ROOT,
    ROAD_MASK_ROOT,
    HISTOGRAM_REFERENCE_ROOT,
    HISTOGRAM_REFERENCE_PATH,
    HISTOGRAM_MATCHING_ROOT
)

from road_damage_detection.config.settings import (
    IMAGE_EXTENSIONS,
    REFERENCE_SAMPLE_SIZE,
    RANDOM_SEED 
)

TRAIN_IMAGE_ROOT = (
    ITERATIVE_IMAGE_ROOT 
    / "train"
)

TRAIN_MASK_ROOT = (
    ROAD_MASK_ROOT 
    / "train"
)

def get_train_images():

    image_paths = [
        path 
        for path in TRAIN_IMAGE_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS 
    ]

    return image_paths 

def sample_road_l(
    image,
    mask,
    random_generator 
):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, _, _ = cv2.split(lab)

    road_mask = mask > 0

    road_l = l[road_mask]

    if len(road_l) == 0:

        return None 

    sample_size = min(
        REFERENCE_SAMPLE_SIZE,
        len(road_l)
    )

    sample = (
        random_generator.choice(
            road_l,
            size = sample_size,
            replace = False 
        )
    )

    return sample 


def build_reference():

    image_paths = get_train_images()

    print(
        f"[INFO] Train images : "
        f"{len(image_paths)}"
    )

    if not image_paths:

        raise FileNotFoundError(
            f"No train images found in : "
            f"{TRAIN_IMAGE_ROOT}"
        )

    random_generator = (
        np.random.default_rng(RANDOM_SEED)
    )

    samples = []

    missing_masks = 0
    empty_masks = 0

    for image_path in tqdm(
        image_paths,
        desc = "Building reference"
    ):

        relative_path = (
            image_path
            .relative_to(
                TRAIN_IMAGE_ROOT 
            )
        )

        mask_path = (
            TRAIN_MASK_ROOT 
            / relative_path 
        ).with_suffix(
            ".png"
        )

        if not mask_path.exists():

            missing_masks += 1 
            continue 

        image = cv2.imread(str(image_path))

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        
        if (
            image is None 
            or mask is None 
        ):

            continue 


        sample = (
            sample_road_l(
                image,
                mask,
                random_generator 
            )
        )


        if sample is None:

            empty_masks += 1 

            continue 


        samples.append(sample)

    if not samples:

        raise RuntimeError(
            "No value road pixels were found."
        )

    reference_l = (
        np.concatenate(
            samples 
        )
    )

    print(
        f"[INFO] Reference pixels : "
        f"{len(reference_l)}"
    )

    print(
        f"[INFO] Missing masks : "
        f"{missing_masks}"
    )

    print(
        f"[INFO] Empty masks : "
        f"{empty_masks}"
    )

    return reference_l 

def save_reference(
    reference_l
):

    HISTOGRAM_REFERENCE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        HISTOGRAM_REFERENCE_PATH,
        reference_l 
    )

    print(
        f"[INFO] Reference saved : "
        f"{HISTOGRAM_REFERENCE_PATH}"
    )

def main():

    reference_l = build_reference()

    save_reference(reference_l)

    print(
        "[INFO] Histogram reference completed."
    )

if __name__ == "__main__":
    main()