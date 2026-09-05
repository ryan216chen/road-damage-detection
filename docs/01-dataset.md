# Dataset

## Dataset Source

本專案主要使用 RDD2022 作為道路損壞偵測資料集。

## Classes

本專案使用以下四個道路損壞類別：

| Class ID | Label |
|---|---|
| 0 | D00 |
| 1 | D10 |
| 2 | D20 |
| 3 | D40 |

## Data Preparation

原始 RDD2022 標註格式為 XML，經轉換後整理為 YOLO 格式。

資料會再進行 train / validation / test 切分，作為後續模型訓練與比較使用。

## Dataset Structure

```text
data/
└── iterative/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

## Notes

後續的 Baseline、Histogram Equalization 與 Histogram Matching 皆使用相同的資料切分，以確保實驗比較一致。