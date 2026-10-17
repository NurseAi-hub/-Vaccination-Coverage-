# 💉 Child Vaccination Coverage & Public Health Analytics

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![scikit-learn](https://img.shields.io/badge/Library-scikit-learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Project Overview

Monitoring and predicting child immunization coverage is a cornerstone of public health policy and preventative epidemiology. This project develops a high-performance Machine Learning pipeline to forecast vaccination coverage rates across diverse demographic categories and vaccine types. By leveraging historical public health surveillance data, this model assists health authorities in identifying coverage gaps and predicting immunization trends.

---

## 📂 Dataset Structure

The dataset (`vaccin-child.csv`) contains public health immunization indicators comprising the following key features:   
* **Year**: The temporal monitoring year of the survey.   
* **Vaccination**: Specific vaccine series or antigen type (e.g., Combined series, DTP, Polio).   
* **Category**: Demographic classification category (e.g., Race and Hispanic origin).   
* **Group**: Sub-population grouping (e.g., All, specific cohorts).   
* **Estimate**: The target variable representing the estimated vaccination coverage rate (percentage).   

---

## ⚙️ Methodology & Technical Pipeline

1. **Data Preprocessing & Robust Cleaning:**
   * Handled column naming conventions safely (including trailing whitespace management).
   * Filtered out missing values in critical health metrics (`Estimate`, `Year`, categories).
2. **Feature Engineering & Encoding:**
   * Applied `LabelEncoder` to transform high-cardinality categorical features (`Vaccination`, `Category`, `Group`) into numerical representations suitable for tree-based algorithms.
3. **Predictive Modeling:**
   * Implemented a `RandomForestRegressor` (`n_estimators=100`), chosen for its robustness against non-linear relationships and capability to model complex demographic interactions.
   * Split data into training and testing sets (80/20 split) with a controlled `random_state` for reproducibility.

---

## 📈 Key Results & Performance

The model achieved exceptional predictive accuracy after incorporating vaccine-specific features:
* **$R^2$ (R-squared) Score:** 0.95 (The model explains 95% of the variance in vaccination coverage rates).
* **Root Mean Squared Error (RMSE):** 2.63% (An exceptionally tight margin of error for population-level health forecasting).

### Feature Importance Insights:
* The model revealed that the specific **Vaccination type** overwhelmingly dictates the baseline coverage rate, while **Category** and **Year** account for demographic shifts and temporal trends over time.

---

## 🛠️ Tech Stack & Requirements

* **Language:** Python 3.x
* **Core Libraries:**
  * `pandas`, `numpy` (Data manipulation & cleaning)
  * `scikit-learn` (Machine Learning pipeline, Label Encoding, Random Forest, Metrics)
  * `matplotlib` (Feature importance data visualization)

---

## 🚀 How to Run the Code

1. Ensure `vaccin-child.csv` is placed in your working directory.   
2. Run the preprocessing and training sequence sequentially in your Jupyter Notebook or Python environment:

```python
# Quick code snippet reference for model training pipeline setup
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('vaccin-child.csv')
# Preprocessing and training steps follow here...
