 # 📊 Advertising Budget & Sales Prediction — LassoCV Model

A machine learning project that predicts sales revenue based on advertising budgets across **TV**, **Radio**, and **Newspaper** channels using a **Lasso Regression with Cross-Validation (LassoCV)** pipeline.

---

## 📁 Dataset

**File:** `Advertising Budget and Sales.csv`

| Column | Description |
|---|---|
| `TV Ad Budget ($)` | Money spent on TV advertising |
| `Radio Ad Budget ($)` | Money spent on Radio advertising |
| `Newspaper Ad Budget ($)` | Money spent on Newspaper advertising |
| `Sales ($)` | Target variable — total sales revenue |

---

## ⚙️ Feature Engineering

A new **interaction feature** is created to capture the combined effect of TV and Radio budgets:

```
TV_Radio_Interaction = TV Ad Budget ($) × Radio Ad Budget ($)
```

This feature significantly improves model accuracy by capturing non-linear relationships.

---

## 🔁 ML Pipeline

The pipeline consists of two sequential steps:

```
Input Features → StandardScaler → LassoCV → Predicted Sales
```

1. **StandardScaler** — Normalizes all features to zero mean and unit variance.
2. **LassoCV** — Automatically selects the best regularization strength (alpha) via 5-fold cross-validation across 100 log-spaced alpha values.

---

## 📈 Exploratory Data Analysis (EDA)

- **Distribution plots** — Histogram + KDE for each feature and the target variable.
- **Correlation Heatmap** — Visualizes pairwise correlations between all features and sales.

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. Run the Notebook

```bash
jupyter notebook gim.ipynb
```

### 3. Predict a Custom Campaign

Use the built-in `predict_new_campaign()` function:

```python
predicted_sales = predict_new_campaign(
    tv_budget=200.0,
    radio_budget=40.0,
    newspaper_budget=15.0
)
print(f"Expected Sales: ${predicted_sales:.2f}")
```

---

## 📊 Model Evaluation

The model is evaluated on a **20% hold-out test set** (80/20 split):

| Metric | Description |
|---|---|
| **R² Score** | Proportion of variance explained by the model |
| **RMSE** | Root Mean Squared Error (lower is better) |
| **Best Alpha** | Optimal regularization value chosen by cross-validation |

Results are printed automatically after training, along with a **Actual vs. Predicted Sales** scatter plot.

---

## 🏗️ Project Structure

```
├── gim.ipynb                        # Main notebook
├── Advertising Budget and Sales.csv # Dataset
└── README.md                        # Project documentation
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LassoCV-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0)

- **NumPy** — Numerical operations
- **Pandas** — Data loading and manipulation
- **Matplotlib / Seaborn** — Data visualization
- **scikit-learn** — ML pipeline, scaling, LassoCV, and metrics
