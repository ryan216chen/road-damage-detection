# Road Damage Detection 

本專案以道路損壞偵測為研究主題，主要使用 RDD2022 資料集，透過SegFormer 與 YOLO 建立影像前處理與道路損壞偵測流程


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

## Installation

```bash
git clone https://github.com/ryan216chen/road-damage-detection.git
cd road-damage-detection

python -m venv .venv
pip install -e .
```