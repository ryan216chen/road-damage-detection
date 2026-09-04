from pathlib import Path
import xml.etree.ElementTree as ET


DATASET_DIR = Path(
    r"D:\road-damage-detection\data\iterative"
)

CLASS_MAP = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3
}


def convert_xml(
    xml_path,
    txt_path
):

    root = ET.parse(
        xml_path
    ).getroot()

    # -------------------------
    # 圖片大小
    # -------------------------

    size = root.find("size")

    width = float(
        size.findtext("width")
    )

    height = float(
        size.findtext("height")
    )

    lines = []

    # -------------------------
    # 每一個 object
    # -------------------------

    for obj in root.findall("object"):

        class_name = obj.findtext("name")

        # 不是我們要的類別就跳過
        if class_name not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[
            class_name
        ]

        bbox = obj.find(
            "bndbox"
        )

        xmin = float(
            bbox.findtext("xmin")
        )

        ymin = float(
            bbox.findtext("ymin")
        )

        xmax = float(
            bbox.findtext("xmax")
        )

        ymax = float(
            bbox.findtext("ymax")
        )

        # -------------------------
        # VOC → YOLO
        # -------------------------

        x_center = (
            (xmin + xmax) / 2
        ) / width

        y_center = (
            (ymin + ymax) / 2
        ) / height

        box_width = (
            xmax - xmin
        ) / width

        box_height = (
            ymax - ymin
        ) / height

        line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        lines.append(
            line
        )

    # -------------------------
    # 建立輸出資料夾
    # -------------------------

    txt_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 就算沒有目標，也建立空 txt
    txt_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def main():

    total = 0

    for split in [
        "train",
        "val"
    ]:

        annotation_dir = (
            DATASET_DIR
            / "annotations"
            / split
        )

        label_dir = (
            DATASET_DIR
            / "labels"
            / split
        )

        for xml_path in annotation_dir.rglob(
            "*.xml"
        ):

            # 例如：
            # Japan/Japan_001.xml
            relative_path = (
                xml_path.relative_to(
                    annotation_dir
                )
            )

            # Japan/Japan_001.txt
            txt_path = (
                label_dir
                / relative_path
            ).with_suffix(
                ".txt"
            )

            convert_xml(
                xml_path,
                txt_path
            )

            total += 1

    print(
        f"轉換完成，共處理 {total} 個 XML"
    )


if __name__ == "__main__":
    main()