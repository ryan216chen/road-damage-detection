from pathlib import Path 
from collections import Counter 

DATA_DIR = Path(r"D:\rdd_project\data\RDD2022")

def main():

    suffix_counts = Counter()

    for file_path in DATA_DIR.rglob("*"):

        if not file_path.is_file():
            continue 

        suffix = file_path.suffix.lower()

        if suffix:
            suffix_counts[suffix] += 1

    print("File extensions:")

    for suffix, count in sorted(
        suffix_counts.items()
    ):
        print(
            f"{suffix} : {count}"
        )


if __name__ == "__main__":
    main()