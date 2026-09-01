from pathlib import Path 
import csv 

FILE_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

PATCH_ROOT = (
    FILE_ROOT 
    / "data"
    / "histogram"
    / "patches"
)


CLASSIFIER_ROOT = (
    FILE_ROOT 
    / "data"
    / "histogram"
    / "classifier"
)

LABEL_CSV = (
    PATCH_ROOT
    / "labels.csv"
)

SPLIT_CSV = (
    CLASSIFIER_ROOT 
    / "split.csv"
)

def load_split():

    split_map = {}

    with open(
        SPLIT_CSV,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            split_map[
                row["source_image"]
            ] = row["split"]

    return split_map 

def load_patch_metadata():

    split_map = load_split()

    patches = []

    with open(
        LABEL_CSV,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            source_image = row["source_image"]

            if source_image not in split_map:
                continue 

            patches.append({
                "source_image" : source_image,
                "patch_name" : row["patch_name"],
                "label" : row["label"],
                "split" : split_map[source_image],
                "x1" : int(row["x1"]),
                "y1" : int(row["y1"]),
                "x2" : int(row["x2"]),
                "y2" : int(row["y2"]),
                "road_ratio" : float(row["road_ratio"])
            })

    return patches 

def main():

    patches = load_patch_metadata()

    print(
        f"[INFO] Patches : "
        f"{len(patches)}"
    )

    for patch in patches[:5]:
        print(patch)

if __name__ == "__main__":
    main()