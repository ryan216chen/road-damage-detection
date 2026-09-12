import cv2
import numpy as np
from tqdm import tqdm

from road_damage_detection.config.paths import (
    ITERATIVE_IMAGE_ROOT
)

from concurrent.futures import ThreadPoolExecutor 


TRAIN_IMAGE_ROOT = (
    ITERATIVE_IMAGE_ROOT
    / "train"
)

MAX_WORKERS = 12


def calculate_brightness(
    image_path
):

    image = cv2.imread(str(image_path))

    if image is None:
        return None 

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel = lab[:, :, 0]

    brightness = l_channel.mean()

    return (
        image_path,
        brightness 
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

    brightness_values = []

    with ThreadPoolExecutor(
        max_workers = MAX_WORKERS
    ) as executor:

        results = executor.map(
            calculate_brightness,
            image_paths 
        )

        for result in tqdm(
            results,
            total=len(image_paths),
            desc="分析圖片亮度"
        ):

            if result is None:
                continue 

            brightness_values.append(result)

    
    values = np.array(
        [
            brightness 
            for _, brightness 
            in brightness_values 
        ]
    )


    print(
        "圖片數量 :",
        len(values)
    )

    print(
        "平均亮度 :",
        values.mean()
    )

    print(
        "亮度標準差 :",
        values.std()
    )

    print(
        "最低亮度 :",
        values.min()
    )

    print(
        "最高亮度 :",
        values.max()
    )

    print(
        "亮度差距 :",
        values.max()
        - values.min()
    )

    print(
    "P10 :",
    np.percentile(
        values,
        10
    )
)

    print(
        "P25 :",
        np.percentile(
            values,
            25
        )
    )

    print(
        "P50 :",
        np.percentile(
            values,
            50
        )
    )

    print(
        "P75 :",
        np.percentile(
            values,
            75
        )
    )

    print(
        "P90 :",
        np.percentile(
            values,
            90
        )
    )


if __name__ == "__main__":
    main()