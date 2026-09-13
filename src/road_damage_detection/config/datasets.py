from road_damage_detection.config.paths import (
    PROJECT_ROOT,
    SEGMENTATION_ROOT,
    HISTOGRAM_EQUALIZATION_ROOT,
    HISTOGRAM_MATCHING_ROOT,
    ITERATIVE_ROOT,
)


DATASET_ROOTS = {
    "baseline": SEGMENTATION_ROOT,
    "equalized": HISTOGRAM_EQUALIZATION_ROOT,
    "matched": HISTOGRAM_MATCHING_ROOT,
    "iterative": ITERATIVE_ROOT,
}


DATASET_CONFIG_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "configs"
)


def get_dataset_root(
    dataset_name
):
    return DATASET_ROOTS[
        dataset_name
    ]


def add_dataset_argument(
    parser
):
    parser.add_argument(
        "--dataset",
        choices=DATASET_ROOTS.keys(),
        required=True
    )


def create_dataset_yaml(
    dataset_name
):
    dataset_root = get_dataset_root(
        dataset_name
    )

    DATASET_CONFIG_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    yaml_path = (
        DATASET_CONFIG_ROOT
        / f"{dataset_name}.yaml"
    )

    content = f"""
path: "{dataset_root.as_posix()}"

train: images/train
val: images/val

names:
  0: D00
  1: D10
  2: D20
  3: D40
"""

    yaml_path.write_text(
        content.strip(),
        encoding="utf-8"
    )

    return yaml_path