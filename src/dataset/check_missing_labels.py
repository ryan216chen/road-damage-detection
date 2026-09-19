from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from tqdm import tqdm

from road_damage_detection.config.paths import (
    ITERATIVE_ROOT
)


MAX_WORKERS = 8


def check_image(
    image_path: Path,
    image_dir: Path,
    label_dir: Path
):

    relative_path = (
        image_path
        .relative_to(image_dir)
    )

    label_path = (
        label_dir
        / relative_path
    ).with_suffix(
        ".txt"
    )

    if not label_path.exists():

        return (
            "missing",
            image_path
        )

    if not label_path.read_text(
        encoding="utf-8"
    ).strip():

        return (
            "empty",
            image_path
        )

    return (
        "labeled",
        image_path
    )


def check_split(
    split: str
):

    image_dir = (
        ITERATIVE_ROOT
        / "images"
        / split
    )

    label_dir = (
        ITERATIVE_ROOT
        / "labels"
        / split
    )

    image_paths = list(
        image_dir.rglob(
            "*.jpg"
        )
    )

    missing_labels = []
    empty_labels = []
    labeled_images = []

    worker = partial(
        check_image,
        image_dir=image_dir,
        label_dir=label_dir
    )

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        results = executor.map(
            worker,
            image_paths
        )

        for (
            status,
            image_path
        ) in tqdm(
            results,
            total=len(image_paths),
            desc=f"Checking {split}"
        ):

            if status == "missing":

                missing_labels.append(
                    image_path
                )

            elif status == "empty":

                empty_labels.append(
                    image_path
                )

            elif status == "labeled":

                labeled_images.append(
                    image_path
                )

    print(
        f"\n[{split}]"
    )

    print(
        f"Images : "
        f"{len(image_paths)}"
    )

    print(
        f"Labels : "
        f"{len(labeled_images)}"
    )

    print(
        f"Missing label file : "
        f"{len(missing_labels)}"
    )

    print(
        f"Empty label file : "
        f"{len(empty_labels)}"
    )


def main():

    for split in [
        "train",
        "val"
    ]:

        check_split(
            split
        )


if __name__ == "__main__":
    main()