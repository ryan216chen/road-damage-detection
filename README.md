# Road Damage Detection

本專案以道路損壞偵測為研究主題，主要使用 RDD2022 資料集，透過 SegFormer 與 YOLO 建立影像前處理與道路損壞偵測流程。

## Pipeline

```text
RDD2022
   ↓
Data Preprocessing
   ↓
Road Segmentation
   ↓
├── Baseline
├── Histogram Equalization
└── Histogram Matching
   ↓
YOLO Training
   ↓
Performance Comparison
```

## Project Structure

```text
road-damage-detection/
├── data/
├── docs/
├── runs/
├── src/
│   └── road_damage_detection/
│       ├── config/
│       ├── preprocessing/
│       └── training/
├── pyproject.toml
└── README.md
```

## Installation

```bash
git clone https://github.com/ryan216chen/road-damage-detection.git
cd road-damage-detection

python -m venv .venv
pip install -e .
```

## Dataset

本專案主要使用 RDD2022 資料集。

資料集不直接包含於 GitHub repository 中，使用前需先準備資料並放入指定的 `data/` 目錄。

詳細的資料處理方式將記錄於 `docs/`。

## Usage

道路區域分割：

```bash
rdd-segment-road
```

套用 road mask：

```bash
rdd-apply-road-mask
```

建立 histogram matching reference：

```bash
rdd-build-reference
```

Histogram equalization：

```bash
rdd-equalize-histogram
```

Histogram matching：

```bash
rdd-match-histogram
```

YOLO training：

```bash
rdd-train --dataset baseline
rdd-train --dataset equalized
rdd-train --dataset matched
```

## Documentation

更詳細的資料處理流程、實驗設計與開發紀錄將整理於 [`docs/`](docs/)。
