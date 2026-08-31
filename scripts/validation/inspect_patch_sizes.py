from pathlib import Path
from collections import Counter
import cv2


FILE_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

IMAGE_ROOT = (
    FILE_ROOT
    / "data"
    / "classifier_dataset"
    / "raw"
    / "test"
)


def main():

    sizes = Counter()

    image_paths = list(
        IMAGE_ROOT.rglob("*.jpg")
    )

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        height, width = (
            image.shape[:2]
        )

        sizes[
            (width, height)
        ] += 1

    print(
        f"Total patches : "
        f"{len(image_paths)}"
    )

    print()
    print(
        "Patch sizes:"
    )

    for (
        width,
        height
    ), count in sizes.most_common():

        print(
            f"{width} x {height} : "
            f"{count}"
        )


if __name__ == "__main__":
    main()