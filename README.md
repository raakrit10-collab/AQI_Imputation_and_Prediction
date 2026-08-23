# Air Quality Index (AQI) Prediction with Missing Data Imputation

An end-to-end framework for urban pollutant time-series imputation, continuous AQI regression, and air quality bucket classification across Indian cities.

---

## 📌 Executive Summary

Air quality monitoring networks frequently suffer from missing sensor data due to technical outages, maintenance, or missing pollutant channels. This project provides a modular deep learning and machine learning pipeline to:

1. **Reconstruct missing pollutant data** using a Deep Bidirectional GRU with Self-Attention, City Embeddings, and Seasonal Temporal Embeddings (**BRATI-Seasonal**).
2. **Predict continuous AQI values** using ML regression ensembles trained on reconstructed multi-pollutant signals.
3. **Classify discrete AQI severity categories** according to official Central Pollution Control Board (CPCB) standards.

---

## 📊 Benchmark Results

### 1. Deep Learning Imputation (BRATI-Seasonal)
- **Validation MSE**: `0.143` (Epoch 3 checkpoint saved via early stopping)
- **Dataset Coverage**: `100%` (29,531 / 29,531 rows reconstructed)
- **City-wise Quality**: Top reconstruction accuracy achieved in Thiruvananthapuram (normalized MAE: `0.147`), Hyderabad (`0.165`), and Visakhapatnam (`0.195`).

### 2. Downstream Regression (Predicting Continuous AQI)
Evaluated on an 80/20 train-test split over 29,531 observations:

| Rank | Model | MAE (AQI points) | RMSE | R² Score | Fit Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest** | **18.90** | **39.60** | **0.9035** | 2.0s |
| 🥈 | **XGBoost** | 19.46 | 39.81 | 0.9024 | 0.4s |
| 🥉 | **MLP Regressor** | 21.35 | 39.99 | 0.9015 | 4.8s |
| 4 | Gradient Boosting | 21.65 | 41.85 | 0.8922 | 4.1s |
| 5 | KNN Regressor | 21.75 | 44.27 | 0.8794 | 0.2s |
| 6 | Ridge Regression | 28.69 | 49.91 | 0.8466 | <0.1s |
| 7 | Linear Regression | 28.69 | 49.91 | 0.8466 | <0.1s |

### 3. Downstream Classification (Predicting AQI Bucket)
Categorized into 6 CPCB buckets (*Good, Satisfactory, Moderate, Poor, Very Poor, Severe*):

| Rank | Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost** | **82.65%** | **0.8255** | **0.8265** | **0.8255** |
| 🥈 | **Random Forest** | 82.46% | 0.8239 | 0.8246 | 0.8232 |
| 🥉 | **Hist Gradient Boosting** | 82.31% | 0.8221 | 0.8231 | 0.8221 |
| 4 | Gradient Boosting | 80.67% | 0.8055 | 0.8067 | 0.8053 |
| 5 | MLP Classifier | 80.24% | 0.8013 | 0.8024 | 0.8010 |
| 6 | KNN Classifier | 77.81% | 0.7763 | 0.7781 | 0.7761 |
| 7 | Logistic Regression | 75.10% | 0.7488 | 0.7510 | 0.7456 |
| 8 | Decision Tree | 74.39% | 0.7450 | 0.7439 | 0.7442 |

---

## 📁 Repository Structure

```
aqi_project/
├── run_pipeline.py                     # Master CLI runner (impute, evaluate, train, or all)
├── train_imputer.py                    # Train deep learning imputer & output imputed dataset
├── evaluate_imputation.py              # Evaluate imputation accuracy & generate diagnostic plots
├── train_models.py                     # Benchmark downstream regression & classification models
├── requirements.txt                    # Project dependencies
├── README.md                           # Project documentation & empirical results
├── src/                                # Modular core package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py                  # Dataset loading, date parsing, CPCB formula, windowing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── imputer.py                  # BRATI-Seasonal PyTorch architecture
│   │   └── ml_models.py                # Regression and classification model factories
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── train_imputation.py         # Imputer training loop & dataset filling
│   │   ├── eval_imputation.py          # Held-out evaluation & plotting pipeline
│   │   └── train_downstream.py         # Downstream model fitting & evaluation pipeline
│   └── utils/
│       ├── __init__.py
│       └── visualization.py            # PDF & plot generation utilities
├── data/
│   └── data.csv                        # Raw multi-city air quality dataset
├── outputs/                            # Model checkpoints and benchmark CSVs
├── figures/                            # Diagnostic plots and comparison charts
└── results/                            # Summary results log
```

---

## 🛠️ Environment Setup

Activate the project environment containing PyTorch, scikit-learn, XGBoost, and Matplotlib:

```bash
source /home/nandhita/aqi-env/bin/activate
```

Alternatively, install required packages into your Python 3.10+ environment:

```bash
pip install -r requirements.txt
```

---

## 🚀 Execution & Usage

### 1. Run Complete Pipeline (One-Command Execution)
Run all stages sequentially:
```bash
python run_pipeline.py --stage all
```

### 2. Stage-by-Stage Execution

#### Step 1: Train Deep Learning Imputer
Train the BRATI-Seasonal model and output `outputs/data_imputed.csv`:
```bash
python train_imputer.py --epochs 50 --batch-size 128 --hidden-size 256
```

#### Step 2: Evaluate Imputation Performance
Calculate held-out metrics and generate PDF time-series plots:
```bash
python evaluate_imputation.py
```

#### Step 3: Train Downstream ML Models
Train and compare 7 regression models and 8 classification models:
```bash
python train_models.py
```

---

## 💡 Methodology & Technical Design

1. **BRATI-Seasonal Imputer**:
   - Uses Bidirectional GRU units to process temporal context in both forward and backward directions.
   - Self-Attention mechanism weighs relevant historical time steps.
   - City and seasonal embeddings capture geographic and intra-annual cyclic variation.
2. **CPCB AQI Formula Integration**:
   - AQI is calculated using official Indian breakpoint sub-indices across 7 key sub-index pollutants (`PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`, `NH3`).
   - Formula sub-indexing is applied strictly to fill missing AQI labels, preserving real-world observations.
3. **Data Leakage Mitigation**:
   - Strict chronological train/validation splitting ensures zero window overlap between training and validation sets.
