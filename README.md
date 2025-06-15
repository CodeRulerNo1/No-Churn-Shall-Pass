
# No Churn Shall Pass

Every lost user means lost revenue. In the crowded fintech space, spotting customers who are about to bail—and winning them back—can make or break growth.
Build a working churn-prediction tool and dashboard demo.

- Generate churn-probability scores for each user.

- Deliver an interactive dashboard that visualizes those scores.
---

## Overview

This project is a churn prediction tool and interactive dashboard for fintech customer data. It predicts the probability of customer churn using both Random Forest and XGBoost models, and visualizes results with Streamlit. The dashboard supports light/dark mode, feature importance, AUC comparison, and lets you upload your own CSV for batch predictions.

---

## Features

- **Upload & Parse:** Upload a CSV of customer data. Handles missing and noisy data.
- **Churn Prediction:** Predicts 30-day churn using Random Forest and XGBoost (AUC-optimized).
- **Visualization:**
  - Churn probability distribution (histograms)
  - Churn vs. retain pie charts
  - Feature importance (pie charts)
  - AUC comparison (pie charts)
  - Top-10 at-risk customers table
- **Download:** Download predictions as CSV.
- **Theme:** Toggle between light and dark mode.
- **Explainability:** Feature importance for both models.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/CodeRulerNo1/No-Churn-Shall-Pass.git
cd No-Churn-Shall-Pass
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Main dependencies:**
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib

### 3. Prepare Models

Train the models using `train.ipynb` or use the provided model files:
- `best_churn_model.pkl` (Random Forest)
- `best_xgb_churn_model.pkl` (XGBoost)

Place these files in the project root.

### 4. Run the Dashboard

```bash
streamlit run app.py
```

---

## Usage

- **Open the dashboard** in your browser (Streamlit will provide a local URL).
- **Upload a CSV** with customer data (must include a `customerID` column).
- **View predictions, plots, and feature importances.**
- **Download** the predictions as a CSV.
- **Toggle light/dark mode** from the sidebar.

---

## Data Format

Your input CSV should include at least these columns:

- `customerID`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `Contract`
- `InternetService`
- `PaymentMethod`
- `OnlineSecurity`
- (other columns are handled but not required for prediction)

---

## Model Training

See `train.ipynb` for:

- Data cleaning, upsampling, and feature engineering
- Model training (Random Forest & XGBoost)
- Hyperparameter tuning (GridSearchCV)
- Model evaluation (AUC, ROC, classification report)
- Saving models with `joblib`

---

## Prediction Approach

- **Random Forest** and **XGBoost** models are trained on engineered features.
- Both models are optimized for AUC-ROC.
- Feature importance is extracted and visualized for both models.
- Predictions are made on uploaded data and visualized in the dashboard.

---

## Visualizations

- **Churn Probability Distribution:** Histogram for each model.
- **Churn vs. Retain:** Pie chart for each model.
- **Feature Importance:** Pie chart for each model.
- **AUC Comparison:** Pie chart for each model.
- **Top-10 At-Risk Customers:** Table in sidebar.

---

## Model Performance


| Model         | AUC   | Test Accuracy |
|---------------|-------|--------------|
| Random Forest | 0.96  | 0.89         |
| XGBoost       | 0.92  | 0.87         |

---

## Customization

- **Theme:** Use the sidebar toggle to switch between light and dark mode.
- **Feature Importance:** Update feature importance values in `app.py` if retraining models.

---





## License

[MIT](https://choosealicense.com/licenses/mit/)

---
## Authors

- [@Ichhabal Singh](https://www.github.com/CodeRulerNo1) (Team Lead)
  - Worked on Model training, integration of model with dashboard and improved dashboard visualizations and functions. 
- [@Anuj Sangwan](https://github.com/aj711)
  - Worked on the design and implementation of dashboard.  
