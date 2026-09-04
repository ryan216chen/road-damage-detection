import shutil

import numpy as np
import torch

from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation
)

from road_damage_detection.config.paths import (
    ITERATIVE_IMAGE_ROOT,
    ROAD_MASK_ROOT
)

from road_damage_detection.config.settings import (
    SEGFORMER_MODEL,
    IMAGE_EXTENSIONS
)


DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


def load_model():

    print(
        f"[INFO] Device : {DEVICE}"
    )

    print(
        f"[INFO] Model : {SEGFORMER_MODEL}"
    )

    processor = (
        AutoImageProcessor
        .from_pretrained(
            SEGFORMER_MODEL
        )
    )

    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            SEGFORMER_MODEL
        )
        .to(DEVICE)
    )

    model.eval()

    return (
        processor,
        model
    )


def get_road_class_id(
    model
):

    label2id = (
        model.config.label2id
    )

    for (
        label,
        class_id
    ) in label2id.items():

        if (
            label
            .strip()
            .lower()
            == "road"
        ):

            print(
                f"[INFO] Road class ID : "
                f"{class_id}"
            )

            return class_id

    raise ValueError(
        "Road class not found."
    )


def segment_road(
    image_path,
    processor,
    model,
    road_class_id
):

    image = (
        Image
        .open(image_path)
        .convert("RGB")
    )

    width, height = (
        image.size
    )

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value
        in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

    segmentation = (
        processor
        .post_process_semantic_segmentation(
            outputs,
            target_sizes=[
                (
                    height,
                    width
                )
            ]
        )[0]
        .cpu()
        .numpy()
    )

    road_mask = (
        segmentation
        == road_class_id
    )

    binary_mask = (
        road_mask
        .astype(np.uint8)
        * 255
    )

    return binary_mask


def get_image_paths():

    image_paths = [
        path
        for path
        in ITERATIVE_IMAGE_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    return image_paths


def process_dataset(
    processor,
    model,
    road_class_id
):

    if ROAD_MASK_ROOT.exists():

        print(
            f"[INFO] Removing existing output : "
            f"{ROAD_MASK_ROOT}"
        )

        shutil.rmtree(
            ROAD_MASK_ROOT
        )

    ROAD_MASK_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = (
        get_image_paths()
    )

    print(
        f"[INFO] Images : "
        f"{len(image_paths)}"
    )

    for image_path in tqdm(
        image_paths,
        desc="Segmenting road"
    ):

        relative_path = (
            image_path
            .relative_to(
                ITERATIVE_IMAGE_ROOT
            )
        )

        output_path = (
            ROAD_MASK_ROOT
            / relative_path
        ).with_suffix(
            ".png"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        road_mask = (
            segment_road(
                image_path,
                processor,
                model,
                road_class_id
            )
        )

        Image.fromarray(
            road_mask
        ).save(
            output_path
        )


def main():

    processor, model = (
        load_model()
    )

    road_class_id = (
        get_road_class_id(
            model
        )
    )

    process_dataset(
        processor,
        model,
        road_class_id
    )

    print(
        "[INFO] Road segmentation completed."
    )


if __name__ == "__main__":
    main()