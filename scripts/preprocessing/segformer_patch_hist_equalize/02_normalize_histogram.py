from pathlib import Path 
import cv2 
import numpy as np 
from tqdm import tqdm 

FILE_ROOT =(
    Path(__file__)
    .resolve()
    .parents[3]
)

IMAGE_ROOT = (
    FILE_ROOT 
    / "data"
    / "iterative"
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
    / "histogram_equalized"
    / "images"
    / "train"
)

def equalize_road(
    image,
    mask 
):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    road_mask = mask > 0

    if not np.any(
        road_mask
    ):
        return image 

    road_l = (
        l[road_mask]
    )

    equalized_l = (
        cv2.equalizeHist(
            road_l.reshape(-1, 1)
        )
        .reshape(-1)
    )

    new_l = l.copy()

    new_l[
        road_mask
    ] = equalized_l 

    equalized_lab = (
        cv2.merge([
            new_l,
            a,
            b 
        ])
    )

    result = cv2.cvtColor(
        equalized_lab,
        cv2.COLOR_LAB2BGR
    )

    result[
        ~road_mask
    ] = 0

    return result 

def main():

    image_paths = list(
        IMAGE_ROOT.rglob("*.jpg")
    )

    print(
        f"[INFO] Images : "
        f"{len(image_paths)}"
    )

    for image_path in tqdm(
        image_paths,
        desc = "Histogram equalization"
    ):

        relative_path = image_path.relative_to(IMAGE_ROOT)

        mask_path = (
            MASK_ROOT
            / relative_path.with_suffix(
                ".png"
            )
        )


        if not mask_path.exists():

            print(
                f"[WARN] Mask not found : "
                f"{mask_path}"
            )

            continue 

        image = cv2.imread(str(image_path))

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            print(
                f"[WARN] Cannot read image"
                f"{image_path}"
            )

            continue 

        if mask is None:

            print(
                f"[WARN] Cannot read mask"
                f"{mask_path}"
            )

            continue 


        result = equalize_road(
            image,
            mask 
        )

        output_path = (
            OUTPUT_ROOT 
            / relative_path 
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(output_path),
            result 
        )

    print(
        f"[INFO] Histogram equalization completed."
    )

if __name__ == "__main__":
    main()