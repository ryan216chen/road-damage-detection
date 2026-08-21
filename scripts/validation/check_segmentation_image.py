from pathlib import Path 

image_dir = Path(r"D:\road-damage-detection\data\road_mask\images")

def main():

    jpg_count = 0

    for image_path in image_dir.rglob("*"):

        if not image_path.is_file():
            continue 

        if image_path.suffix.lower() == ".jpg":
            jpg_count += 1

    print(jpg_count)

if __name__ == "__main__":
    main()