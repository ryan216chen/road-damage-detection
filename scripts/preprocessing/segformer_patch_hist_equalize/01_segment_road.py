from pathlib import Path 
import numpy as np 
import torch 
from PIL import Image 
from tqdm import tqdm 
import shutil

from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation
)

FILE_ROOT = Path(__file__).resolve().parents[3]

INPUT_ROOT = FILE_ROOT / "data" / "iterative" / "images"

OUTPUT_IMAGE_ROOT = FILE_ROOT / "data" / "segformer_road" / "images"

OUTPUT_MASK_ROOT = OUTPUT_IMAGE_ROOT.parent / "masks"

if OUTPUT_IMAGE_ROOT.exists():
    shutil.rmtree(OUTPUT_IMAGE_ROOT)
if OUTPUT_MASK_ROOT.exists():
    shutil.rmtree(OUTPUT_MASK_ROOT)


MODEL_NAME = (
    "nvidia/"
    "segformer-b5-finetuned-ade-640-640"
)

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

def load_model():
    print(
        f"[INFO] Device : "
        f"{DEVICE}"
    )

    print(
        f"[INFO] Model : "
        f"{MODEL_NAME}"
    )

    processor = (
        AutoImageProcessor
        .from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            MODEL_NAME
        )
        .to(DEVICE)
    )

    model.eval()

    return (
        processor,
        model 
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

    original_width, original_height = (
        image.size 
    )

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }


    with torch.inference_mode():

        outputs = model(**inputs)

    segmentation = (
        processor
        .post_process_semantic_segmentation(
            outputs,
            target_sizes=[
                (
                    original_height,
                    original_width 
                )
            ],
        )[0]
    )

    segmentation = (
        segmentation
        .cpu()
        .numpy()
    )

    road_mask = (
        segmentation == road_class_id 
    )

    image_array = np.array(
        image 
    )

    road_image = np.zeros_like(
        image_array 
    )

    road_image[road_mask] = image_array[road_mask]

    binary_mask = (
        road_mask.astype(
            np.uint8
        )
        * 255
    )

    return (
        road_image,
        binary_mask 
    )


def process_dataset(
    processor,
    model,
    road_class_id
):

    image_paths = [
        path 
        for path in INPUT_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() == ".jpg"
        )
    ]

    print(
        f"[INFO] Images : "
        f"{len(image_paths)}"
    )

    for image_path in tqdm(
        image_paths,
        desc="Segmenting roads"
    ):

        relative_path = (
            image_path
            .relative_to(
                INPUT_ROOT
            )
        )

        output_image_path = (
            OUTPUT_IMAGE_ROOT
            / relative_path 
        )

        output_mask_path = (
            OUTPUT_MASK_ROOT
            / relative_path 
            .with_suffix(".png")
        )

        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)

        road_image, road_mask = (
            segment_road(
                image_path,
                processor,
                model,
                road_class_id 
            )
        )


        Image.fromarray(
            road_image
        ).save(
            output_image_path 
        )

        Image.fromarray(
            road_mask
        ).save(
            output_mask_path 
        )


def get_road_class_id(
    model 
):

    label2id = model.config.label2id 

    for label, class_id in label2id.items():

        if label.strip().lower() == "road":
            print(
                f"[INFO] Road Class : "
                f"{class_id}"
            )

            return class_id 

    raise ValueError(
        "Road class was not found "
        "in model labels."
    )

def main():

    processor, model = load_model()

    road_class_id = get_road_class_id(model)

    process_dataset(
        processor,
        model,
        road_class_id 
    )

    print("[INFO] Road segmentation completed.")

if __name__ == "__main__":
    main()