import shutil 
import cv2 
import numpy as np 
from tqdm import tqdm 

from road_damage_detection.config.paths import (
    ITERATIVE_IMAGE_ROOT,
    ITERATIVE_LABEL_ROOT,
    ROAD_MASK_ROOT,
    SEGMENTATION_IMAGE_ROOT,
    SEGMENTATION_LABEL_ROOT 
)

from road_damage_detection.config.settings import (
    IMAGE_EXTENSIONS 
)

def get_image_paths():

    image_paths = [
        path 
        for path in ITERATIVE_IMAGE_ROOT.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS 
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

            shutil.rmtree(root)

        else:

            root.unlink()

def apply_road_mask(
    image,
    mask 
):

    road_mask = mask > 0 

    result = np.zeros_like(image)

    result[road_mask] = image[road_mask]

    return result 

def process_dataset():

    clear_output_root(SEGMENTATION_IMAGE_ROOT)
    clear_output_root(SEGMENTATION_LABEL_ROOT)

    SEGMENTATION_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_LABEL_ROOT.mkdir(parents=True, exist_ok=True)

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

    empty_masks = 0

    for image_path in tqdm(
        image_paths,
        desc = "Applying road masks"
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
        ).with_suffix(".png")


        if not mask_path.exists():

            raise FileNotFoundError(
                f"Mask not found : "
                f"{mask_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:

            raise RuntimeError(
                f"Failed ot read image : "
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

        if not np.any(mask > 0):

            empty_masks += 1 

        result = apply_road_mask(
            image,
            mask 
        )

        output_image_path = (
            SEGMENTATION_IMAGE_ROOT 
            / relative_path 
        )

        output_image_path.parent.mkdir(parents=True, exist_ok=True)

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
            .with_suffix(".txt")
        )

        source_label_path = (
            ITERATIVE_LABEL_ROOT 
            / label_relative_path
        )

        output_label_path = (
            SEGMENTATION_LABEL_ROOT 
            / label_relative_path 
        )

        if not source_label_path.exists():

            raise FileNotFoundError(
                f"Label not found : "
                f"{source_label_path}"
            )

        output_label_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(
            source_label_path,
            output_label_path 
        )

    print(
        f"[INFO] Empty masks : "
        f"{empty_masks}"
    )

def main():

    process_dataset()

    print(
        "[INFO] Segformer baseline dataset completed."
    )

if __name__ == "__main__":
    main()