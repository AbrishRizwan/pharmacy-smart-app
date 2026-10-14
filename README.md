# 💊 Automated Pharmaceutical Sales Forecasting Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Engineering-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An automated machine learning pipeline designed to forecast daily medicine sales across multiple pharmaceutical categories, helping pharmacies prevent stockouts and minimize inventory expiration.

---

## 📌 Problem Statement

Pharmaceutical inventory management requires a delicate balance between meeting patient demand and avoiding capital lockup:
* **Understocking:** Can result in life-threatening treatment delays for critical medications.
* **Overstocking:** Leads to high holding costs and waste due to batch expiration dates.

Traditional static reorder methods fail to account for weekly demand patterns, monthly seasonality, and autocorrelation. This project builds a multi-category forecasting pipeline to predict future daily unit sales from historical transactional data.

---

## 🛠️ Data Pipeline & Feature Engineering

The pipeline processes daily sales records across distinct therapeutic drug categories (e.g., analgesics, anti-inflammatories, antibiotics).

1. **Time-Series Decomposition & Resampling:**
   - Aggregated raw sales transactions into regular daily and weekly frequency intervals.
   - Handled non-trading days and imputed zero-demand periods.
2. **Feature Engineering:**
   - **Lag Features:** 1-day, 7-day, 14-day, and 30-day historical sales lags to capture demand momentum.
   - **Rolling Window Statistics:** 7-day and 14-day rolling means and standard deviations to capture moving trends and volatility.
   - **Calendar & Seasonal Signals:** Day of week, day of month, month, quarter, and weekend flags.
3. **Model Training:**
   - Trained category-specific `RandomForestRegressor` models to capture non-linear demand interactions and temporal shifts without requiring strict stationarity assumptions.

---

## 📊 Evaluation & Metrics

The models are evaluated using out-of-time validation splits to mirror production deployment:

| Metric | Purpose | Clinical/Operational Significance |
| :--- | :--- | :--- |
| **MAE (Mean Absolute Error)** | Quantifies average error in unit sales | Informs baseline reorder buffer size |
| **RMSE (Root Mean Squared Error)** | Penalizes large forecasting errors heavily | Identifies unexpected demand spikes |
| **R² Score** | Measures proportion of variance explained | Validates model fit against a naive baseline |

---

## 📁 Repository Structure

```text
├── data/
│   └── pharma_sales_data.csv          # Raw daily sales dataset
├── models/
│   └── random_forest_models.pkl       # Serialized trained category models
├── notebooks/
│   └── sales_forecasting_pipeline.ipynb # EDA, feature engineering, and model training
├── src/
│   ├── data_preprocessing.py          # Data cleaning and lag feature generation
│   ├── train.py                       # Automated training pipeline script
│   └── predict.py                     # Inference script for future sales forecasting
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
````
## How to Run Locally
```bash
# 1. Clone the repository
git clone [https://github.com/AbrishRizwan/pharma-sales-forecasting.git](https://github.com/AbrishRizwan/pharma-sales-forecasting.git)

# 2. Navigate to project directory
cd pharma-sales-forecasting

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the automated training and forecasting pipeline
python src/train.py

```

## Author

[Abrish Rizwan](https://github.com/AbrishRizwan)
