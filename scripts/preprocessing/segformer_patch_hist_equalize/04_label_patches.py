from pathlib import Path
import csv
import cv2


FILE_ROOT = (
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


PATCH_CSV = (
    FILE_ROOT
    / "data"
    / "patch_dataset"
    / "patches.csv"
)


LABEL_CSV = (
    FILE_ROOT
    / "data"
    / "patch_dataset"
    / "labels.csv"
)


MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 900


def load_patches():

    image_patches = {}

    with open(
        PATCH_CSV,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            source_image = (
                row["source_image"]
            )

            patch = {
                "patch_name":
                    row["patch_name"],

                "row":
                    int(row["row"]),

                "col":
                    int(row["col"]),

                "x1":
                    int(row["x1"]),

                "y1":
                    int(row["y1"]),

                "x2":
                    int(row["x2"]),

                "y2":
                    int(row["y2"]),

                "road_ratio":
                    float(row["road_ratio"])
            }

            if (
                source_image
                not in image_patches
            ):

                image_patches[
                    source_image
                ] = []

            image_patches[
                source_image
            ].append(
                patch
            )

    return image_patches


def load_completed_images():

    completed_images = set()

    if not LABEL_CSV.exists():

        return completed_images

    with open(
        LABEL_CSV,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            completed_images.add(
                row["source_image"]
            )

    return completed_images


def get_display_scale(
    image
):

    height, width = (
        image.shape[:2]
    )

    width_scale = (
        MAX_DISPLAY_WIDTH
        / width
    )

    height_scale = (
        MAX_DISPLAY_HEIGHT
        / height
    )

    scale = min(
        width_scale,
        height_scale,
        1.0
    )

    return scale


def resize_for_display(
    image,
    scale
):

    if scale == 1.0:

        return image.copy()

    height, width = (
        image.shape[:2]
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )

    return resized


def draw_interface(
    image,
    patches,
    selected,
    scale,
    image_index,
    total_images
):

    preview = resize_for_display(
        image,
        scale
    )

    for index, patch in enumerate(
        patches
    ):

        x1 = int(
            patch["x1"]
            * scale
        )

        y1 = int(
            patch["y1"]
            * scale
        )

        x2 = int(
            patch["x2"]
            * scale
        )

        y2 = int(
            patch["y2"]
            * scale
        )

        if index in selected:

            color = (
                0,
                0,
                255
            )

            text = "CRACK"

        else:

            color = (
                0,
                255,
                0
            )

            text = "NORMAL"

        cv2.rectangle(
            preview,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            color,
            2
        )

        cv2.putText(
            preview,
            text,
            (
                x1 + 5,
                y1 + 22
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        road_ratio_text = (
            f"{patch['road_ratio']:.0%}"
        )

        cv2.putText(
            preview,
            road_ratio_text,
            (
                x1 + 5,
                y1 + 45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )

    crack_count = len(
        selected
    )

    candidate_count = len(
        patches
    )

    info_text = (
        f"Image "
        f"{image_index}/{total_images}"
        f"   "
        f"Candidates: {candidate_count}"
        f"   "
        f"Crack: {crack_count}"
    )

    cv2.rectangle(
        preview,
        (
            0,
            0
        ),
        (
            min(
                preview.shape[1],
                750
            ),
            40
        ),
        (
            0,
            0,
            0
        ),
        -1
    )

    cv2.putText(
        preview,
        info_text,
        (
            10,
            27
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            255,
            255,
            255
        ),
        2
    )

    help_text = (
        "Click: toggle crack | "
        "N: save + next | "
        "R: reset | "
        "Q: quit"
    )

    text_y = (
        preview.shape[0]
        - 15
    )

    cv2.rectangle(
        preview,
        (
            0,
            text_y - 28
        ),
        (
            min(
                preview.shape[1],
                850
            ),
            preview.shape[0]
        ),
        (
            0,
            0,
            0
        ),
        -1
    )

    cv2.putText(
        preview,
        help_text,
        (
            10,
            text_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            255,
            255,
            255
        ),
        1
    )

    return preview


def label_image(
    image_path,
    patches,
    image_index,
    total_images
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"[WARN] Cannot read image : "
            f"{image_path}"
        )

        return (
            "skip",
            None
        )

    scale = get_display_scale(
        image
    )

    selected = set()

    window_name = (
        "Patch Labeling"
    )

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if (
            event
            != cv2.EVENT_LBUTTONDOWN
        ):

            return

        original_x = int(
            x / scale
        )

        original_y = int(
            y / scale
        )

        for index, patch in enumerate(
            patches
        ):

            inside_x = (
                patch["x1"]
                <= original_x
                < patch["x2"]
            )

            inside_y = (
                patch["y1"]
                <= original_y
                < patch["y2"]
            )

            if (
                inside_x
                and inside_y
            ):

                if index in selected:

                    selected.remove(
                        index
                    )

                else:

                    selected.add(
                        index
                    )

                break

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_AUTOSIZE
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    while True:

        preview = draw_interface(
            image,
            patches,
            selected,
            scale,
            image_index,
            total_images
        )

        cv2.imshow(
            window_name,
            preview
        )

        key = (
            cv2.waitKey(30)
            & 0xFF
        )

        if (
            key == ord("n")
            or key == ord("N")
        ):

            labels = []

            for index, patch in enumerate(
                patches
            ):

                if index in selected:

                    label = "crack"

                else:

                    label = "normal"

                labels.append({
                    "patch_name":
                        patch[
                            "patch_name"
                        ],

                    "label":
                        label,

                    "row":
                        patch[
                            "row"
                        ],

                    "col":
                        patch[
                            "col"
                        ],

                    "x1":
                        patch[
                            "x1"
                        ],

                    "y1":
                        patch[
                            "y1"
                        ],

                    "x2":
                        patch[
                            "x2"
                        ],

                    "y2":
                        patch[
                            "y2"
                        ],

                    "road_ratio":
                        patch[
                            "road_ratio"
                        ]
                })

            cv2.destroyWindow(
                window_name
            )

            return (
                "save",
                labels
            )

        if (
            key == ord("r")
            or key == ord("R")
        ):

            selected.clear()

        if (
            key == ord("q")
            or key == ord("Q")
            or key == 27
        ):

            cv2.destroyWindow(
                window_name
            )

            return (
                "quit",
                None
            )


def save_labels(
    writer,
    file,
    source_image,
    labels
):

    for item in labels:

        writer.writerow({
            "source_image":
                source_image,

            "patch_name":
                item[
                    "patch_name"
                ],

            "label":
                item[
                    "label"
                ],

            "row":
                item[
                    "row"
                ],

            "col":
                item[
                    "col"
                ],

            "x1":
                item[
                    "x1"
                ],

            "y1":
                item[
                    "y1"
                ],

            "x2":
                item[
                    "x2"
                ],

            "y2":
                item[
                    "y2"
                ],

            "road_ratio":
                item[
                    "road_ratio"
                ]
        })

    file.flush()


def main():

    if not PATCH_CSV.exists():

        raise FileNotFoundError(
            f"Patch CSV not found : "
            f"{PATCH_CSV}"
        )

    image_patches = (
        load_patches()
    )

    completed_images = (
        load_completed_images()
    )

    all_images = list(
        image_patches.items()
    )

    remaining_images = [
        (
            source_image,
            patches
        )
        for (
            source_image,
            patches
        )
        in all_images
        if (
            source_image
            not in completed_images
        )
    ]

    print(
        f"[INFO] Total images : "
        f"{len(all_images)}"
    )

    print(
        f"[INFO] Completed : "
        f"{len(completed_images)}"
    )

    print(
        f"[INFO] Remaining : "
        f"{len(remaining_images)}"
    )

    if not remaining_images:

        print(
            "[INFO] All images "
            "have been labeled."
        )

        return

    LABEL_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
        LABEL_CSV.exists()
        and LABEL_CSV.stat().st_size > 0
    )

    with open(
        LABEL_CSV,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "source_image",
            "patch_name",
            "label",
            "row",
            "col",
            "x1",
            "y1",
            "x2",
            "y2",
            "road_ratio"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:

            writer.writeheader()

        total_remaining = len(
            remaining_images
        )

        for index, (
            source_image,
            patches
        ) in enumerate(
            remaining_images,
            start=1
        ):

            image_path = (
                IMAGE_ROOT
                / source_image
            )

            print()
            print(
                f"[INFO] "
                f"{index}/"
                f"{total_remaining}"
            )

            print(
                f"[INFO] Image : "
                f"{source_image}"
            )

            print(
                f"[INFO] Candidates : "
                f"{len(patches)}"
            )

            action, labels = (
                label_image(
                    image_path,
                    patches,
                    index,
                    total_remaining
                )
            )

            if action == "quit":

                print()
                print(
                    "[INFO] Labeling stopped."
                )

                print(
                    "[INFO] Saved labels : "
                    f"{LABEL_CSV}"
                )

                break

            if action == "skip":

                continue

            if action == "save":

                save_labels(
                    writer,
                    file,
                    source_image,
                    labels
                )

                crack_count = sum(
                    item["label"]
                    == "crack"
                    for item in labels
                )

                normal_count = (
                    len(labels)
                    - crack_count
                )

                print(
                    f"[INFO] Saved : "
                    f"crack={crack_count}, "
                    f"normal={normal_count}"
                )

    cv2.destroyAllWindows()

    print()
    print(
        f"[INFO] Labels saved to : "
        f"{LABEL_CSV}"
    )


if __name__ == "__main__":
    main()