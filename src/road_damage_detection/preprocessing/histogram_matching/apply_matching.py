import shutil 
import cv2 
import numpy as np 
from skimage.exposure import match_histograms 
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
    IMAGE_EXTENSIONS 
)

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

    clear_output_root(MATCHED_IMAGE_ROOT)
    clear_output_root(MATCHED_LABEL_ROOT) 

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

            raise FileNotFoundError(
                f"Mask not found : "
                f"{mask_path}"
            )

        image = cv2.imread(str(image_path))

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            raise RuntimeError(
                f"Failed to read mask : "
                f"{mask_path}"
            )

        if image is None:

            raise RuntimeError(
                f"Failed to read image : "
                f"{image_path}"
            )

        

        if not np.any(
            mask > 0
        ):

            empty_masks += 1
            result = image 
        
        else:

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



        label_relative_path = (
            relative_path
            .with_suffix(".txt")
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
