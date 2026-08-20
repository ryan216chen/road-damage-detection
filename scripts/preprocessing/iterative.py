from pathlib import Path 
import xml.etree.ElementTree as ET 
import shutil 

import numpy as np 
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedShuffleSplit 
)

data_dir = Path(r"D:\rdd_dataset\RDD2022")
output_dir = Path(r"D:\road-damage-detection\data\iterative")

CLASSES = [
    "D00",
    "D10",
    "D20",
    "D40"
]

COUNTRIES = [
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
]

def get_labels(xml_path):

    root = ET.parse(
        xml_path
    ).getroot()

    labels = set()

    for obj in root.findall("object"):

        name = obj.findtext("name")

        if name in CLASSES:
            labels.add(name)

    return labels 

def copy_dataset(
    records,
    split 
):

    for record in records:

        xml_path = record["xml"]
        image_path = record["image"]
        country = record["country"]

        output_image_dir = (
            output_dir 
            / "images"
            / split 
            / country 
        )

        output_xml_dir = (
            output_dir 
            / "annotations"
            / split 
            / country 
        )

        output_image_dir.mkdir(
            parents=True,
            exist_ok=True 
        )

        output_xml_dir.mkdir(
            parents=True,
            exist_ok=True 
        )

        shutil.copy2(
            image_path,
            output_image_dir / image_path.name 
        )

        shutil.copy2(
            xml_path,
            output_xml_dir / xml_path.name 
        )


def main():

    records = []

    for country in COUNTRIES:

        train_dir = (
            data_dir 
            / country 
            / "train"
        )

        xml_dir = (
            train_dir 
            / "annotations"
            / "xmls"
        )

        image_dir = (
            train_dir 
            / "images"
        )

        image_dict = {}

        for image_path in image_dir.rglob("*"):

            if(
                image_path.is_file()
                and image_path.suffix.lower()
                in [".jpg"]
            ):
                image_dict[image_path.stem] = image_path 

        for xml_path in xml_dir.rglob("*"):

            image_path = image_dict.get(xml_path.stem)

            if image_path is None:
                print("找不到圖片 : ", xml_path.name)

                continue 

            labels = get_labels(xml_path)

            records.append(
                {
                    "xml" : xml_path,
                    "image" : image_path,
                    "country" : country,
                    "labels" : labels 
                }
            )

    print(f"圖片總數 : {len(records)}")

    #multi-label matrix 

    y = []

    for record in records:

        row = []

        for class_name in CLASSES:

            row.append(
                int(
                    class_name
                    in record["labels"]
                )
            )

        #negative

        row.append(
            int(
                len(record["labels"]) == 0
            )
        )            


        for country in COUNTRIES:

            row.append(
                int(
                    record["country"] == country 
                )
            )

        y.append(row)

    y = np.array(y)

    # 8 : 2 split

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits = 1,
        test_size = 0.2,
        random_state = 37
    )

    train_index, val_index = next(
        splitter.split(
            np.zeros(
                len(records)
            ),
            y 
        )
    )

    train_records = [
        records[i] 
        for i in train_index 
    ]

    val_records = [
        records[i]
        for i in val_index 
    ]

    print(
        "Train : ",
        len(train_records)
    )

    print(
        "Val : ",
        len(val_records)
    )

    if output_dir.exists():

        shutil.rmtree(
            output_dir 
        )

    print("開始建立 Train...")

    copy_dataset(
        train_records,
        "train"
    )

    print("開始建立 Val...")

    copy_dataset(
        val_records,
        "val"
    )

    print()
    print("資料集建立完成 : ", output_dir)

if __name__ == "__main__":
    main()