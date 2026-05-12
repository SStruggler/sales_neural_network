# 🏪 Retail Sales Prediction using Neural Networks

> **A complete, production-ready ML project** — end-to-end pipeline, Power BI dashboard, and interactive Streamlit app.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project builds a deep learning pipeline to **predict retail sales amounts** from the Superstore dataset. It combines:

- **Descriptive Analytics** — Power BI dashboard (what happened in the past)
- **Predictive Analytics** — Neural network deployed via Streamlit (what will happen next)

### What It Predicts
Given a customer segment, product category, discount level, region, and date, the model predicts the **Expected Sales Amount ($)**.

---

## 🏗️ Project Architecture

```
sales-neural-network-project/
│
├── data/
│   ├── raw/                        # Original Superstore CSV
│   └── processed/                  # Cleaned data, Power BI export, predictions
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb   # Interactive EDA, distributions, correlations
│   └── 02_model_prototyping.ipynb  # Training, evaluation, loss curves
│
├── models/
│   ├── sales_model.keras           # Trained ANN (Keras format)
│   └── scaler.joblib               # Fitted StandardScaler (prevents data leakage)
│
├── dashboard/
│   └── POWERBI_BLUEPRINT.md        # 5-page Power BI dashboard specifications + DAX
│
├── app/
│   └── app.py                      # Interactive Streamlit prediction app
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Full feature engineering pipeline
│   └── train_pipeline.py           # ANN architecture, training, evaluation
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Neural Network Architecture

```
Input Layer (20 features)
        ↓
Dense(128, ReLU) → BatchNormalization → Dropout(0.3)
        ↓
Dense(64, ReLU) → Dropout(0.2)
        ↓
Dense(1, Linear)   ← Regression output (unbounded)
```

**Compilation:**
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error
- Metrics: Mean Absolute Error

**Regularization:**
- EarlyStopping (patience=10, monitor=val_loss)
- ReduceLROnPlateau (factor=0.5, patience=5)
- ModelCheckpoint (saves best weights only)

---

## ⚙️ Feature Engineering

| Feature | Type | Description |
|---------|------|-------------|
| `Quantity` | Numerical | Number of items ordered |
| `Discount` | Numerical | Discount fraction (0.0–1.0) |
| `Profit` | Numerical | Transaction profit |
| `discount_percentage` | Derived | Discount × 100 |
| `promo_active` | Binary | 1 if discount > 0 |
| `month`, `year`, `day_of_week` | Temporal | Extracted from Order Date |
| `is_weekend` | Binary | 1 if Sat/Sun |
| `month_sin/cos` | Cyclical | Seasonality encoding |
| `day_sin/cos` | Cyclical | Weekly pattern encoding |
| `Region_*` | OHE | One-hot encoded region (drop_first) |
| `Segment_*` | OHE | One-hot encoded segment |
| `Category_*` | OHE | One-hot encoded product category |

---

## 📊 Model Performance (Test Set)

| Metric | Value |
|--------|-------|
| MAE | ~$143 |
| RMSE | ~$579 |
| R² | ~0.43 |

> **Note:** The Superstore dataset has high sales variance (orders range from $1 to $23,000+), making exact prediction inherently difficult. The model captures general trends well. R² can be improved with more historical data or additional features like customer lifetime value.

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourname/sales-neural-network-project.git
cd sales-neural-network-project
pip install -r requirements.txt
```

### 2. Add Your Data
Place `sales.csv` (Superstore dataset) into `data/raw/`.

### 3. Train the Model
```bash
python src/train_pipeline.py
```
This runs preprocessing → training → evaluation → saves artifacts.

### 4. Launch Streamlit App
```bash
streamlit run app/app.py
```

### 5. Explore Notebooks
```bash
jupyter notebook notebooks/01_eda_and_cleaning.ipynb
```

---

## 📊 Power BI Dashboard

Connect Power BI to `data/processed/powerbi_clean_sales.csv`.

**5 Pages:**
1. **Sales Overview** — Executive KPIs, monthly trends
2. **Regional Performance** — Map-based geographic analysis
3. **Product Analysis** — Pareto chart, margin distribution
4. **Customer Segments** — Consumer vs Corporate vs Home Office
5. **Predicted Sales Trends** — Neural network forecast vs actual

See `dashboard/POWERBI_BLUEPRINT.md` for full DAX measures and visual specifications.

---

## 🛠️ Module Reference

### `src/preprocessing.py`
| Function | Description |
|----------|-------------|
| `load_raw_data()` | Load CSV from disk |
| `clean_data()` | Remove duplicates, parse dates, fill nulls |
| `engineer_datetime_features()` | Month, year, cyclical encoding |
| `engineer_business_features()` | Discount %, promo flag |
| `encode_categoricals()` | OHE with drop_first |
| `split_and_scale()` | Train/test split + StandardScaler (fit on train only) |
| `export_for_powerbi()` | Export enriched CSV for BI |
| `run_preprocessing_pipeline()` | End-to-end entry point |

### `src/train_pipeline.py`
| Function | Description |
|----------|-------------|
| `build_model()` | Define Keras ANN architecture |
| `get_callbacks()` | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |
| `train_model()` | Train with validation split |
| `evaluate_model()` | MAE, RMSE, R² on test set |
| `plot_training_curves()` | Loss & MAE vs epoch |
| `plot_residuals()` | 4-panel diagnostic plots |
| `export_predictions_for_powerbi()` | Actual vs Predicted CSV |
| `run_training_pipeline()` | Full train + eval + export |

---

## 📁 Outputs Generated

| File | Description |
|------|-------------|
| `models/sales_model.keras` | Trained neural network |
| `models/scaler.joblib` | Fitted StandardScaler for inference |
| `data/processed/powerbi_clean_sales.csv` | Enriched dataset for Power BI |
| `data/processed/predictions.csv` | Actual vs Predicted for Power BI Page 5 |
| `data/processed/training_curves.png` | Loss / MAE curve plots |
| `data/processed/residual_diagnostics.png` | 4-panel residual analysis |
| `data/processed/feature_names.txt` | Ordered feature list for Streamlit inference |

---

## 💡 Why This Project Stack?

| Tool | Role |
|------|------|
| **Pandas** | Data wrangling and feature engineering |
| **Scikit-learn** | Preprocessing, splitting, metrics |
| **TensorFlow/Keras** | Neural network definition and training |
| **Joblib** | Scaler serialization |
| **Streamlit** | Interactive prediction web app |
| **Power BI** | Enterprise descriptive analytics dashboard |
| **Matplotlib/Seaborn** | Training visualization and EDA |

---

## 📖 Skills Demonstrated

- ✅ End-to-end ML pipeline design
- ✅ Feature engineering (cyclical encoding, OHE, scaling)
- ✅ Neural network regression with Keras
- ✅ Preventing data leakage (scaler fit on train only)
- ✅ Callback-based training regularization
- ✅ Model evaluation (MAE, RMSE, R²)
- ✅ Residual diagnostic analysis
- ✅ Model serialization and deployment
- ✅ Interactive Streamlit ML app
- ✅ Power BI dashboard design and DAX
- ✅ Business analytics thinking

---

## 📄 License

MIT — free to use, modify, and distribute with attribution.

---

*Built as a portfolio project demonstrating the full spectrum from raw data to business intelligence and predictive deployment.*
