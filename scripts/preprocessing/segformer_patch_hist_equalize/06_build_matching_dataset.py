from pathlib import Path 
import cv2 
import numpy as np 
import shutil 

from skimage.exposure import match_histograms 

from load_patch_metadata import load_patch_metadata

FILE_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

IMAGE_ROOT = (
    FILE_ROOT 
    / "data"
    / "histogram"
    / "segformer_road"
    / "images"
    / "train"
)

MASK_ROOT = (
    FILE_ROOT 
    / "data"
    / "histogram"
    / "segformer_road"
    / "masks"
    / "train"
)

OUTPUT_ROOT = (
    FILE_ROOT 
    / "data"
    / "histogram"
    / "classifier"
    / "matched"
)

def build_reference(
    patches 
):

    source_images = {
        item["source_image"]
        for item in patches
        if item["split"] == "train" 
    }

    samples = []

    for source_image in source_images:

        image = cv2.imread(str(IMAGE_ROOT / source_image))
        mask = cv2.imread(
            str(MASK_ROOT / Path(source_image).with_suffix(".png")),
            cv2.IMREAD_GRAYSCALE
        )


        if (image is None
            or mask is None
        ):
            continue 

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        l, _, _ = cv2.split(lab)

        road_l = l[mask > 0]

        if len(road_l) == 0:
            continue 
    
        sample_size = min(5000, len(road_l))

        sample = np.random.choice(
            road_l,
            sample_size,
            replace=False 
        )

        samples.append(sample)

    return np.concatenate(samples)



def match_image(
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

    matched_l = match_histograms(
        road_l,
        reference_l 
    )

    l[road_mask] = (
        matched_l
        .clip(0, 255)
        .astype(np.uint8)
    )


    result = cv2.cvtColor(
        cv2.merge([
            l,
            a,
            b 
        ]),
        cv2.COLOR_LAB2BGR
    )

    result[~road_mask] = 0 

    return result 

def main():

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    patches = load_patch_metadata()

    reference_l = build_reference(patches)

    for item in patches:

        image_path = (
            IMAGE_ROOT 
            / item["source_image"]
        )      

        mask_path = (
            MASK_ROOT 
            / item["source_image"]
        ).with_suffix(".png")

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



        matched_image = match_image(
            image,
            mask,
            reference_l 
        )

        patch = matched_image[
            item["y1"] : item["y2"],
            item["x1"] : item["x2"]
        ]

        output_path = (
            OUTPUT_ROOT 
            / item["split"]
            / item["label"]
            / Path(item["source_image"]).parent 
            / item["patch_name"]
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(output_path),
            patch 
        )

if __name__ == "__main__":
    main()