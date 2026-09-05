# Road Damage Detection 

本專案以道路損壞偵測為研究主題，主要使用 RDD2022 資料集，透過Segformer 與 YOLO 建立影像錢處理與道路損壞偵測流程

## Pipeline

RDD2022 

$$
\downarrow
$$ 

Iterative Strafication 

$$
\downarrow
$$

Road Segmentation

$$
\downarrow 
$$ 

├── Baseline
├── Histogram Equalization
└── Histogram Matching

$$
\downarrow
$$ 

YOLO Training 

$$
\downarrow
$$ 

Performance Comparison 

## Installation

```bash
git clone https://github.com/ryan216chen/road-damage-detection.git
cd road-damage-detection

python -m venv .venv
pip install -e .
```