# Customer Churn Prediction

Predict customer churn using Kaggle’s Telco dataset with Scikit-learn.

## Dataset
- Source: [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Features: `tenure`, `MonthlyCharges`, `Contract`, etc.
- Target: `Churn` (Yes/No)

## Methodology
- Preprocessed data: Handled missing values, encoded categorical variables, scaled numerical features.
- Trained logistic regression model, achieving 82% accuracy.
- Analyzed feature importance, identifying `tenure` and `Contract` as key predictors.

## Results
- **Accuracy**: 0.8211, indicating strong overall performance.
- **F1-Score (Churn=1)**: 0.6400, reflecting good performance on the minority class despite class imbalance (~73% No, 27% Yes).
- **Key Features**: `tenure`, `Contract_Month-to-month`, and `InternetService_Fiber optic` are top predictors (see `feature_importance.png`).
- **Metrics**: Full details in `model_metrics.txt`.

## Setup
```bash
conda create -n machine-learning-3-11 python=3.11
conda activate machine-learning-3-11
pip install -r requirements.txt
jupyter notebook