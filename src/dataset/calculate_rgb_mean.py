from concurrent.futures import ThreadPoolExecutor

import cv2 
import numpy as np 
from tqdm import tqdm 

from road_damage_detection.config.paths import (
    ITERATIVE_IMAGE_ROOT
)

TRAIN_IMAGE_ROOT = (
    ITERATIVE_IMAGE_ROOT
    / "train"
)

MAX_WORKERS = 12

def calculate_image_statistics(
    image_path 
):

    image = cv2.imread(str(image_path))

    if image is None:
        return None 

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    rgb = (
        rgb.astype(np.float64)
        / 255.0
    )

    pixels = rgb.reshape(
        -1,
        3
    )

    pixel_sum = pixels.sum(
        axis = 0
    )

    pixel_square_sum = (
        pixels ** 2
    ).sum(
        axis = 0
    )

    pixel_count = pixels.shape[0]

    return (
        pixel_sum,
        pixel_square_sum,
        pixel_count 
    )


def main():

    image_paths = [
        image_path
        for image_path in TRAIN_IMAGE_ROOT.rglob("*")
        if (
            image_path.is_file()
            and image_path.suffix.lower() == ".jpg"
        )
    ]

    print(
        "圖片數量 :",
        len(image_paths)
    )

    total_sum = np.zeros(
        3,
        dtype = np.float64 
    )

    total_square_sum = np.zeros(
        3,
        dtype = np.float64 
    )

    total_pixels = 0

    with ThreadPoolExecutor(
        max_workers = MAX_WORKERS
    ) as executor:

        results = executor.map(
            calculate_image_statistics,
            image_paths 
        )

        for result in tqdm(
            results,
            total = len(image_paths),
            desc = "計算RGB mean/std"
        ):

            if result is None:
                continue 

            (
                pixel_sum,
                pixel_square_sum,
                pixel_count 
            ) = result


            total_sum += (
                pixel_sum 
            )

            total_square_sum += (
                pixel_square_sum 
            )

            total_pixels += (
                pixel_count 
            )

    rgb_mean = (total_sum / total_pixels)

    rgb_variance = (
        total_square_sum 
        / total_pixels 
        - rgb_mean ** 2
    )

    rgb_variance = np.maximum(
        rgb_variance,
        0
    )

    rgb_std = np.sqrt(rgb_variance)

    print()
    print("RGB Mean")

    print(
        "R :",
        rgb_mean[0]
    )

    print(
        "G :",
        rgb_mean[1]
    )

    print(
        "B :",
        rgb_mean[2]
    )

    print()
    print("RGB Std")

    print(
        "R :",
        rgb_std[0]
    )

    print(
        "G :",
        rgb_std[1]
    )

    print(
        "B :",
        rgb_std[2]
    )

if __name__ == "__main__":
    main()