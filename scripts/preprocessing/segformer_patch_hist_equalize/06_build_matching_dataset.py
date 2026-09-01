from pathlib import Path 
import cv2 

from load_patch_metadata import (load_patch_metadata)

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

OUTPUT_ROOT = (
    FILE_ROOT 
    / "data"
    / "histogram"
    / "classifier"
    / "matched"
)

def main():

    patches = load_patch_metadata()

    for item in patches:

        image_path = (
            IMAGE_ROOT 
            / item["source_image"]
        )      

        image = cv2.imread(str(image_path))

        if image is None:
            continue 

        matched_image = image 

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