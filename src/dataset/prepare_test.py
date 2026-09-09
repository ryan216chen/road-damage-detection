from pathlib import Path 
import shutil 

from road_damage_detection.config.paths import TEST_IMAGE_ROOT 

RDD2022_ROOT = Path(r"D:\rdd_dataset\RDD2022")

COUNTRIES = [
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
]

def main():

    if TEST_IMAGE_ROOT.exists():
        shutil.rmtree(TEST_IMAGE_ROOT)

    total = 0

    for country in COUNTRIES:

        source_root = (
            RDD2022_ROOT 
            / country 
            / "test"
            / "images"
        )

        if not source_root.exists():

            print(
                f"[WARNING] Not Found : "
                f"{source_root}"
            )

            continue 

        output_root = (
            TEST_IMAGE_ROOT
            / country 
        )

        output_root.mkdir(parents=True, exist_ok=True)

        for image_path in source_root.rglob("*.jpg"):

            shutil.copy2(
                image_path,
                output_root 
                / image_path.name 
            )

            total += 1 

    print(
        f"[INFO] Test images : "
        f"{total}"
    )

    print(
        f"[INFO] Output : "
        f"{TEST_IMAGE_ROOT}"
    )

if __name__ == "__main__":
    main()

