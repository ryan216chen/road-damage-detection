from pathlib import Path
from collections import (
    defaultdict,
    Counter
)
import csv
import random
import shutil


FILE_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)


PATCH_ROOT = (
    FILE_ROOT
    / "data"
    / "patch_dataset"
)


LABEL_CSV = (
    PATCH_ROOT
    / "labels.csv"
)


RAW_PATCH_ROOT = (
    PATCH_ROOT
    / "raw"
)


EQUALIZED_PATCH_ROOT = (
    PATCH_ROOT
    / "equalized"
)


OUTPUT_ROOT = (
    FILE_ROOT
    / "data"
    / "classifier_dataset"
)


SPLIT_CSV = (
    OUTPUT_ROOT
    / "split.csv"
)


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


def load_labels():

    image_labels = defaultdict(
        list
    )

    with open(
        LABEL_CSV,
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

            patch_name = (
                row["patch_name"]
            )

            label = (
                row["label"]
                .strip()
                .lower()
            )

            if label not in (
                "crack",
                "normal"
            ):
                continue

            image_labels[
                source_image
            ].append({
                "patch_name":
                    patch_name,

                "label":
                    label
            })

    return image_labels


def split_images(
    image_labels
):

    groups = defaultdict(
        list
    )

    for (
        source_image,
        patches
    ) in image_labels.items():

        has_crack = any(
            patch["label"] == "crack"
            for patch in patches
        )

        source_path = Path(
            source_image
        )

        if len(
            source_path.parts
        ) > 1:

            group_name = (
                source_path.parts[0]
            )

        else:

            group_name = "root"

        key = (
            group_name,
            has_crack
        )

        groups[
            key
        ].append(
            source_image
        )

    split_map = {}

    random_generator = (
        random.Random(
            SEED
        )
    )

    for images in groups.values():

        images = (
            images.copy()
        )

        random_generator.shuffle(
            images
        )

        count = len(
            images
        )

        train_count = int(
            count
            * TRAIN_RATIO
        )

        val_count = int(
            count
            * VAL_RATIO
        )

        train_images = (
            images[
                :train_count
            ]
        )

        val_images = (
            images[
                train_count:
                train_count
                + val_count
            ]
        )

        test_images = (
            images[
                train_count
                + val_count:
            ]
        )

        for source_image in (
            train_images
        ):

            split_map[
                source_image
            ] = "train"

        for source_image in (
            val_images
        ):

            split_map[
                source_image
            ] = "val"

        for source_image in (
            test_images
        ):

            split_map[
                source_image
            ] = "test"

    return split_map


def get_patch_path(
    root,
    source_image,
    patch_name
):

    source_path = Path(
        source_image
    )

    patch_path = (
        root
        / source_path.parent
        / patch_name
    )

    return patch_path


def copy_patch(
    source_path,
    output_root,
    split,
    label,
    source_image,
    patch_name
):

    source_parent = (
        Path(source_image)
        .parent
    )

    output_path = (
        output_root
        / split
        / label
        / source_parent
        / patch_name
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        source_path,
        output_path
    )


def save_split_csv(
    image_labels,
    split_map
):

    with open(
        SPLIT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "source_image",
            "split",
            "patch_count",
            "crack_count",
            "normal_count"
        ])

        for (
            source_image,
            patches
        ) in image_labels.items():

            crack_count = sum(
                patch["label"]
                == "crack"
                for patch in patches
            )

            normal_count = (
                len(patches)
                - crack_count
            )

            writer.writerow([
                source_image,
                split_map[
                    source_image
                ],
                len(patches),
                crack_count,
                normal_count
            ])


def build_dataset(
    image_labels,
    split_map
):

    statistics = Counter()

    missing_count = 0

    for (
        source_image,
        patches
    ) in image_labels.items():

        split = split_map[
            source_image
        ]

        for patch in patches:

            patch_name = (
                patch["patch_name"]
            )

            label = (
                patch["label"]
            )

            raw_path = (
                get_patch_path(
                    RAW_PATCH_ROOT,
                    source_image,
                    patch_name
                )
            )

            equalized_path = (
                get_patch_path(
                    EQUALIZED_PATCH_ROOT,
                    source_image,
                    patch_name
                )
            )

            if (
                not raw_path.exists()
                or
                not equalized_path.exists()
            ):

                print(
                    f"[WARN] Missing patch : "
                    f"{patch_name}"
                )

                missing_count += 1

                continue

            copy_patch(
                source_path=raw_path,
                output_root=(
                    OUTPUT_ROOT
                    / "raw"
                ),
                split=split,
                label=label,
                source_image=source_image,
                patch_name=patch_name
            )

            copy_patch(
                source_path=equalized_path,
                output_root=(
                    OUTPUT_ROOT
                    / "equalized"
                ),
                split=split,
                label=label,
                source_image=source_image,
                patch_name=patch_name
            )

            statistics[
                (
                    split,
                    label
                )
            ] += 1

    return (
        statistics,
        missing_count
    )


def print_statistics(
    image_labels,
    split_map,
    statistics,
    missing_count
):

    image_statistics = Counter(
        split_map.values()
    )

    print()
    print(
        "===== Images ====="
    )

    print(
        f"Total : "
        f"{len(image_labels)}"
    )

    for split in (
        "train",
        "val",
        "test"
    ):

        print(
            f"{split:5} : "
            f"{image_statistics[split]}"
        )

    print()
    print(
        "===== Patches ====="
    )

    for split in (
        "train",
        "val",
        "test"
    ):

        crack_count = (
            statistics[
                (
                    split,
                    "crack"
                )
            ]
        )

        normal_count = (
            statistics[
                (
                    split,
                    "normal"
                )
            ]
        )

        total = (
            crack_count
            + normal_count
        )

        print(
            f"{split:5} : "
            f"total={total}, "
            f"crack={crack_count}, "
            f"normal={normal_count}"
        )

    print()
    print(
        f"Missing patches : "
        f"{missing_count}"
    )


def main():

    if not LABEL_CSV.exists():

        raise FileNotFoundError(
            f"Labels CSV not found : "
            f"{LABEL_CSV}"
        )

    if OUTPUT_ROOT.exists():

        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "[INFO] Loading labels..."
    )

    image_labels = (
        load_labels()
    )

    print(
        f"[INFO] Labeled images : "
        f"{len(image_labels)}"
    )

    split_map = (
        split_images(
            image_labels
        )
    )

    save_split_csv(
        image_labels,
        split_map
    )

    (
        statistics,
        missing_count
    ) = build_dataset(
        image_labels,
        split_map
    )

    print_statistics(
        image_labels,
        split_map,
        statistics,
        missing_count
    )

    print()
    print(
        "[INFO] Classifier "
        "dataset completed."
    )

    print(
        f"[INFO] Output : "
        f"{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()