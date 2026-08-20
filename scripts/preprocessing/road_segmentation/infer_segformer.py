from pathlib import Path 
import numpy as np 
import torch 
import torch.nn.functional as F 
from PIL import Image 

from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation
)

MODEL_NAME = (
    "nvidia/"
    "segformer-b2-finetuned-cityscapes-1024-1024"
)

INPUT_DIR = Path(r"D:\road-damage-detection\data\iterative\images")

OUTPUT_DIR = Path(r"D:\road-damage-detection\data\road_mask")

IMAGE_SUFFIXES = {".jpg"}

def load_model():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device : {device}")

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

    def predict_mask(
    image_path,
    processor,
    model,
    device
    ):

        image = (
            Image
            .open(image_path)
            .convert("RGB")
        )

        width, height = image.size

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value
            in inputs.items()
        }

        with torch.no_grad():

            outputs = model(
                **inputs
            )

        logits = outputs.logits

        # SegFormer 輸出的 segmentation
        # 尺寸通常比原圖小
        # 所以 resize 回原圖尺寸
        logits = F.interpolate(
            logits,
            size=(
                height,
                width
            ),
            mode="bilinear",
            align_corners=False
        )

        mask = (
            logits
            .argmax(dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        return mask


def save_mask(
    mask,
    output_path
    ):

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        mask_image = (
            Image.fromarray(
                mask
            )
        )

        mask_image.save(
            output_path
        )


def main():

    processor, model, device = (
        load_model()
    )

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
        f"Images : {len(image_paths)}"
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
        ).with_suffix(".png")

        mask = predict_mask(
            image_path,
            processor,
            model,
            device
        )

        save_mask(
            mask,
            output_path
        )

        print(
            f"[{index}/{len(image_paths)}] "
            f"{relative_path}"
        )


if __name__ == "__main__":
    main()