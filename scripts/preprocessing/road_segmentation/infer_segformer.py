from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation
)


# ========================
# Config
# ========================

MODEL_NAME = (
    "nvidia/"
    "segformer-b5-finetuned-cityscapes-1024-1024"
)

INPUT_DIR = Path(
    r"D:\road-damage-detection\data\iterative\images"
)

OUTPUT_DIR = Path(
    r"D:\road-damage-detection\data\road_mask\images"
)

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png"
}


def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device : {device}"
    )

    print(
        f"Loading model : {MODEL_NAME}"
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
        .to(device)
    )

    model.eval()

    return (
        processor,
        model,
        device
    )


def get_road_mask(
    image,
    processor,
    model,
    device
):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = (
        inputs["pixel_values"]
        .to(device)
    )

    with torch.inference_mode():

        outputs = model(
            pixel_values=pixel_values
        )

    logits = outputs.logits

    # SegFormer 輸出解析度較小
    # 放大回原圖尺寸
    logits = F.interpolate(
        logits,
        size=(
            image.height,
            image.width
        ),
        mode="bilinear",
        align_corners=False
    )

    prediction = (
        logits
        .argmax(dim=1)[0]
        .cpu()
        .numpy()
    )

    road_id = (
        model.config.label2id["road"]
    )

    road_mask = (
        prediction == road_id
    )

    return road_mask


def apply_mask(
    image,
    road_mask
):

    image_array = np.array(
        image
    )

    masked_image = (
        image_array.copy()
    )

    # 非 road pixel 變黑
    masked_image[
        ~road_mask
    ] = 0

    return Image.fromarray(
        masked_image
    )


def process_image(
    image_path,
    output_path,
    processor,
    model,
    device
):

    image = (
        Image.open(
            image_path
        )
        .convert("RGB")
    )

    road_mask = get_road_mask(
        image=image,
        processor=processor,
        model=model,
        device=device
    )

    masked_image = apply_mask(
        image=image,
        road_mask=road_mask
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    masked_image.save(
        output_path,
        quality=95
    )


def main():

    (
        processor,
        model,
        device
    ) = load_model()

    image_paths = [
        path
        for path
        in INPUT_DIR.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_SUFFIXES
        )
    ]

    print(
        f"Total images : "
        f"{len(image_paths)}"
    )

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        relative_path = (
            image_path.relative_to(
                INPUT_DIR
            )
        )

        output_path = (
            OUTPUT_DIR
            / relative_path
        )

        process_image(
            image_path=image_path,
            output_path=output_path,
            processor=processor,
            model=model,
            device=device
        )

        print(
            f"[{index}/{len(image_paths)}] "
            f"{relative_path}"
        )


if __name__ == "__main__":
    main()