# 💊 Smart Pharmacy: Medicine Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red)

A Streamlit demo for medicine demand forecasting. Per-category Random Forest models predict daily unit sales from calendar features; the app compares predicted demand with stock levels. Inventory, POS and analytics pages use sample data.

🔗 **Live demo:** https://pharmacy-smart-app-mpvpjkjmerseesf8bl7kmz.streamlit.app/

## 📌 Problem
Understocking delays treatment, while overstocking causes holding costs and expiry waste. This project explores whether daily unit sales of medicine categories can be forecast from calendar information alone.

## 🛠️ Approach
- **Data:** Kaggle pharmaceutical sales dataset (`salesdaily.csv`), daily sales for 8 ATC medicine categories
- **Features:** year, month, and weekday (one-hot encoded)
- **Model:** one `RandomForestRegressor` per medicine category
- **Evaluation:** time-based 80/20 split (train on earlier dates, test on later dates), compared with a baseline that always predicts the training-set average

## 📊 Results
| Medicine | MAE | Baseline MAE | RMSE | R² |
|---|---|---|---|---|
| M01AB | 2.51 | 2.19 | 3.19 | -0.30 |
| M01AE | 1.83 | 1.74 | 2.44 | -0.16 |
| N02BA | 1.82 | 1.83 | 2.35 | -0.46 |
| N02BE | 10.46 | 12.31 | 14.20 | 0.19 |
| N05B | 3.82 | 3.38 | 4.77 | -0.29 |
| N05C | 0.92 | 0.83 | 1.26 | -0.31 |
| R03 | 6.05 | 5.81 | 8.70 | -0.13 |
| R06 | 1.97 | 1.90 | 2.62 | -0.08 |

**Key finding:** with calendar features only, the model beats the baseline for just one category (N02BE). For the others it does not outperform simply predicting the average. Daily sales appear to depend mostly on recent sales history, which this version does not use.

## ⚠️ Limitations & Next Steps
- Calendar-only features; no lag or rolling-window features yet
- The app multiplies a single-day estimate by the forecast period instead of forecasting each day
- Next step: add lag/rolling features and time-series cross-validation
- Inventory, POS, expiry and analytics pages use sample data and were built with AI-assisted coding

## 📁 Repository Structure
```
├── app.py                 # Streamlit app
├── medicine_models.pkl    # Trained per-category models
├── model_columns.pkl      # Feature columns
├── pharmacy_model_training.ipynb  # Model training notebook
└── requirements.txt
```

## ▶️ Run Locally
```bash
git clone https://github.com/AbrishRizwan/pharmacy-smart-app.git
cd pharmacy-smart-app
pip install -r requirements.txt
streamlit run app.py
```
