from pathlib import Path 
import random 
import csv 
import cv2 
import shutil 

FILE_ROOT = Path(__file__).resolve().parents[3]

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
    / "mask"
)

OUTPUT_ROOT = (
    FILE_ROOT
    / "data"
    / "selected_patches"
)

if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CSV_PATH = (
    OUTPUT_ROOT 
    / "selected_patches.csv"
)

GRID = 4
MIN_ROAD_RATIO = 0.7
SAMPLE_IMAGES = 50

def get_patch(
    image,
    row,
    col 
):

    height, width = (
        image.shape[:2]
    )

    patch_height = (
        height // GRID 
    )

    patch_width = (
        width // GRID 
    )

    y1 = (
        row * patch_height 
    )

    y2 = (
        height 
        if row == GRID - 1
        else (row + 1) * patch_height 
    )

    x1 = (
        col * patch_width 
    )

    x2 = (
        width 
        if col == GRID - 1
        else (col + 1) * patch_width 
    )

    return (
        image[y1:y2, x1:x2],
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

            patch, _, _, _, _ = (
                get_patch(
                    mask,
                    row,
                    col 
                )
            )

            road_ratio = (
                patch > 0
            ).mean()

            if (
                road_ratio 
                >= MIN_ROAD_RATIO 
            ):

                candidates.append(
                    (
                        row,
                        col,
                        road_ratio 
                    )
                )

    return candidates 
    

def select_patches(
    image_path,
    mask_path 
):  
    image = cv2.imread(
        str(image_path)
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    candidates = (
        get_candidates(
            mask 
        )
    )

    selected = set()

    height, width = (
        image.shape[:2]
    )

    patch_height = height // GRID 

    patch_width = width // GRID 

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if (
            event != cv2.EVENT_LBUTTONDOWN
        ):
            return 

        col = min(x // patch_width, GRID - 1)
        row = min(y // patch_height, GRID - 1)

        valid_cells = [
            (r, c)
            for r, c, _ in candidates 
        ]

        if (row, col) not in candidates:
            return 

        if (row, col) in selected:

            selected.remove(
                (row, col)
            )

        else:

            selected.add(
                (row, col)
            )

    window_name = image_path.name 

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )


    cv2.setMouseCallback(
        window_name,
        mouse_callback 
    )

    while True:

        preview = image.copy()

        for (row, col, road_ratio) in candidates:

            (
                _,
                x1,
                y1,
                x2,
                y2 
            ) = get_patch(
                image,
                row,
                col 
            )

            if (row, col) in selected:

                color = (0, 255, 255)

            else:

                color = (0, 255, 0)


            cv2.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                color,
                3
            )


            cv2.putText(
                preview,
                f"{road_ratio:.0%}",
                (
                    x1 + 5,
                    y1 + 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow(
            window_name,
            preview 
        )

        key = (
            cv2.waitKey(30) & 0xFF
        )


        if key == ord("n"):
            break 

        if key == ord("q"):
            cv2.destroyAllWindows()

            return None 

    
