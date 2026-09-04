import shutil 
import cv2 
import numpy as np 
from skimage.exposure import match_histograms 
from tqdm import tqdm 

from road_damage_detection.config.paths import (
    ITERATIVE_IMAGE_ROOT,
    ROAD_MASK_ROOT,
    HISTOGRAM_REFERENCE_PATH,
    MATCHED_IMAGE_ROOT 
)

from road_damage_detection.config.settings import (
    IMAGE_EXTENSIONS 
)

def get_image_paths():

    image_paths = [
        path 
        for path in ITERATIVE_IMAGE_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS 
    ]

    return image_paths 


def apply_matching(
    image,
    mask,
    reference_l 
):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB 
    )

    l, a, b = cv2.split(lab)

    road_mask = mask > 0

    road_l = l[road_mask]

    if len(road_l) == 0:
        return image 

    matched_l = (
        match_histograms(
            road_l,
            reference_l 
        )
    )

    matched_l = (
        matched_l
        .clip(
            0,
            255
        )
        .astype(np.uint8)
    )

    result_l = l.copy()

    result_l[road_mask] = matched_l 

    matched_lab = cv2.merge(
        [
            result_l,
            a,
            b 
        ]
    )

    result = cv2.cvtColor(
        matched_lab,
        cv2.COLOR_LAB2BGR 
    )

    return result 



def process_dataset(
    reference_l 
):

    if MATCHED_IMAGE_ROOT.exists():

        print(
            f"[INFO] Removing existing output : "
            f"{MATCHED_IMAGE_ROOT}"
        )

        if MATCHED_IMAGE_ROOT.is_dir():

            shutil.rmtree(MATCHED_IMAGE_ROOT)

        else:

            MATCHED_IMAGE_ROOT.unlink()

    MATCHED_IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True 
    )

    image_paths = get_image_paths()

    print(
        f"[INFO] Images : "
        f"{len(image_paths)}"
    )

    if not image_paths:

        raise FileNotFoundError(
            f"No images found in : "
            f"{ITERATIVE_IMAGE_ROOT}"
        )

    
    missing_masks = 0
    empty_masks = 0

    for image_path in tqdm(
        image_paths,
        desc = "Histogram matching"
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

        if not np.any(
            mask > 0
        ):

            empty_masks += 1
            continue 

        result = apply_matching(
            image,
            mask,
            reference_l 
        )

        output_path = (
            MATCHED_IMAGE_ROOT 
            / relative_path 
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True 
        )

        cv2.imwrite(
            str(output_path),
            result 
        )

    print(
        f"[INFO] Missing masks : "
        f"{missing_masks}"
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

    print(
        f"[INFO] Loading reference : "
        f"{HISTOGRAM_REFERENCE_PATH}"
    )

    reference_l = np.load(HISTOGRAM_REFERENCE_PATH)

    print(
        f"[INFO] Reference pixels : "
        f"{len(reference_l)}"
    )

    process_dataset(reference_l)

    print(
        f"[INFO] Histogram matching completed."
    )

if __name__ == "__main__":
    main()
