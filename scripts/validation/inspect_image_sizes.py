from pathlib import Path 
from collections import Counter 

import cv2 

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

IMAGE_DIR = (
    PROJECT_ROOT 
    / "data"
    / "iterative"
    / "images"
)


def main():

    size_counts = Counter()

    total_images = 0
    failed_images = 0

    for image_path in IMAGE_DIR.rglob("*.jpg"):

        image = cv2.imread(str(image_path))
        if image is None:

            print(
                f"failed to read: "
                f"{image_path}"
            )

            failed_images += 1
            continue 

        height, width = (
            image.shape[:2]
        )

        size_counts[
            (width, height)
        ] += 1 

        total_images += 1

    print()
    print(
        f"total images : "
        f"{total_images}"
    )

    print(
        f"resolution types : "
        f"{len(size_counts)}"
    )

    print()
    print(
        "image resolutions:"
    )

    for (width, height), count in size_counts.most_common():

        print(
            f"{width} x {height}"
            f" : {count}"
        )

if __name__ == "__main__":
    main()