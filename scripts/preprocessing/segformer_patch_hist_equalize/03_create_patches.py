from pathlib import Path 
import csv 
import cv2 
import shutil 

FILE_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

RAW_IMAGE_ROOT = (
    FILE_ROOT 
    / "data"
    / "segformer_road"
    / "images"
    / "train"
)

EQUALIZED_IMAGE_ROOT = (
    FILE_ROOT 
    / "data"
    / "histogram_equalized"
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
    / "patch_dataset"
)

RAW_OUTPUT_ROOT = (
    OUTPUT_ROOT 
    / "raw"
)

EQUALIZED_OUTPUT_ROOT = (
    OUTPUT_ROOT 
    / "equalized"
)

CSV_PATH = (
    OUTPUT_ROOT 
    / "patches.csv"
)

GRID = 4
MIN_ROAD_RATIO = 0.7


def get_patch(
    image,
    row,
    col 
):

    height, width = image.shape[:2]

    patch_height = height // GRID 

    patch_width = width // GRID 

    y1 = row * patch_height 

    y2 = (
        height 
        if row == GRID - 1
        else (row + 1) * patch_height 
    )

    x1 = col * patch_width 

    x2 = (
        width
        if col == GRID - 1
        else (col + 1) * patch_width 
    )

    patch = (
        image[
            y1:y2,
            x1:x2 
        ]
    )

    return (
        patch,
        x1,
        y1,
        x2,
        y2 
    )

def get_candidates(
    mask 
):

    candidates = []

    for row in range(GRID):

        for col in range(GRID):

            (
                patch,
                _,
                _,
                _,
                _ 
            ) = get_patch(
                mask,
                row,
                col 
            )

            road_ratio = (
                patch > 0
            ).mean()

            if (
                road_ratio >= MIN_ROAD_RATIO
            ):

                candidates.append(
                    (
                        row,
                        col,
                        road_ratio 
                    )
                )

    return candidates 


def save_patches(
    raw_image,
    equalized_image,
    candidates,
    relative_path,
    writer 
):

    for row, col, road_ratio in candidates:

        (
            raw_patch,
            x1,
            y1,
            x2,
            y2
        ) = get_patch(
            raw_image,
            row,
            col 
        )


        (
            equalized_patch,
            _,
            _,
            _,
            _ 
        ) = get_patch(
            equalized_image,
            row,
            col 
        )

        patch_name = (
            f"{relative_path.stem}"
            f"_r{row}_c{col}.jpg"
        )

        raw_output_dir = (
            RAW_OUTPUT_ROOT 
            / relative_path.parent 
        )

        equalized_output_dir = (
            EQUALIZED_OUTPUT_ROOT 
            / relative_path.parent 
        )

        raw_output_dir.mkdir(parents=True, exist_ok=True)

        equalized_output_dir.mkdir(parents=True, exist_ok=True)

        raw_output_path = (
            raw_output_dir 
            / patch_name 
        )

        equalized_output_path = (
            equalized_output_dir
            / patch_name 
        )

        cv2.imwrite(
            str(raw_output_path),
            raw_patch 
        )

        cv2.imwrite(
            str(equalized_output_path),
            equalized_patch 
        )

        writer.writerow([
            relative_path.as_posix(),
            patch_name,
            row,
            col,
            x1,
            y1,
            x2,
            y2,
            road_ratio 
        ])


def process_image(
    raw_image_path,
    writer
):

    relative_path = (
        raw_image_path
        .relative_to(
            RAW_IMAGE_ROOT
        )
    )

    equalized_image_path = (
        EQUALIZED_IMAGE_ROOT
        / relative_path
    )

    mask_path = (
        MASK_ROOT
        / relative_path.with_suffix(
            ".png"
        )
    )

    if not equalized_image_path.exists():
        return

    if not mask_path.exists():
        return

    raw_image = cv2.imread(
        str(raw_image_path)
    )

    equalized_image = cv2.imread(
        str(equalized_image_path)
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if (
        raw_image is None
        or equalized_image is None
        or mask is None
    ):
        return

    candidates = get_candidates(
        mask
    )

    save_patches(
        raw_image,
        equalized_image,
        candidates,
        relative_path,
        writer
    )

def main():

    if OUTPUT_ROOT.exists():

        shutil.rmtree(
            OUTPUT_ROOT
        )

    RAW_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    EQUALIZED_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = list(
        RAW_IMAGE_ROOT.rglob(
            "*.jpg"
        )
    )

    print(
        f"[INFO] Images : "
        f"{len(image_paths)}"
    )

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "source_image",
            "patch_name",
            "row",
            "col",
            "x1",
            "y1",
            "x2",
            "y2",
            "road_ratio"
        ])

        for raw_image_path in image_paths:

            process_image(
                raw_image_path,
                writer
            )

    print(
        "[INFO] Patch creation completed."
    )


if __name__ == "__main__":
    main()