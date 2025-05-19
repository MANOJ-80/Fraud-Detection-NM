# 💳 AI-Powered Credit Card Fraud Detection 

A machine learning solution for real-time identification of fraudulent credit card transactions, leveraging ensemble modeling techniques and explainable AI.

---

## Project Overview

This system addresses credit card fraud (estimated $32+ billion annual losses) using supervised learning models trained on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). The solution provides:

- Real-time fraud probability scoring
- Model interpretability features
- Comprehensive performance analytics

### Technical Highlights

**Model Architecture:**
- Random Forest (baseline)
- SMOTE-enhanced Random Forest (handling class imbalance)
- Gradient Boosting variants (XGBoost, LightGBM, CatBoost)
- AdaBoost (adaptive boosting)

**Key Innovations:**
- Synthetic Minority Over-sampling (SMOTE) for 492:284,807 class ratio
- SHAP values for transaction-level explainability
- Dynamic thresholding based on transaction risk profiles

---

## Dataset Specifications

**Source:** [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
**Characteristics:**
- 284,807 transactions (492 fraudulent)
- 30 features (28 PCA components + Amount + Class)
- Highly skewed distribution (0.172% fraud prevalence)

**Feature Engineering:**
- V1-V28: Principal components from original features
- Amount: Standard scaled (μ=0, σ=1)
- Time: Excluded from final models

---

## Technical Implementation

**Core Stack:**
- Python 3.9+
- Scikit-learn 1.2+
- XGBoost 1.7, LightGBM 3.3, CatBoost 1.2
- Imbalanced-learn 0.10 (SMOTE implementation)
- SHAP 0.42 (model interpretability)

**Analytics:**
- Matplotlib 3.7, Seaborn 0.12 (visualizations)
- Pandas 2.0, NumPy 1.24 (data processing)

**Application Layer:**
- Streamlit 1.22 (web interface)
- Joblib 1.2 (model serialization)

---

## Model Performance Comparison

### Evaluation Metrics Summary

| Model               | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | ROC-AUC | False Negatives | False Positives |
|---------------------|-------------------|----------------|-------------------|---------|-----------------|-----------------|
| SMOTE RandomForest  | 0.85              | 0.84           | 0.84              | 0.918   | 16              | 15              |
| Normal RandomForest | 0.94              | 0.82           | 0.87              | 0.908   | 18              | 5               |
| XGBoost             | 0.92              | 0.81           | 0.86              | 0.903   | 19              | 7               |
| LightGBM            | 0.29              | 0.51           | 0.37              | 0.754   | 48              | 124             |
| CatBoost            | 0.95              | 0.83           | 0.89              | 0.913   | 17              | 4               |
| AdaBoost            | 0.71              | 0.73           | 0.72              | 0.867   | 26              | 30              |

**Key Observations:**
1. **CatBoost** demonstrates the best balance of precision (0.95) and recall (0.83) among all models
2. **SMOTE RandomForest** shows the highest ROC-AUC (0.918) with strong recall
3. **LightGBM** underperforms significantly in fraud detection (precision: 0.29)
4. **Normal RandomForest** achieves the lowest false positives (5) while maintaining good recall
5. All models maintain perfect precision (1.00) for legitimate transactions (Class 0)
   
**Visual Analytics:**
- ROC curve comparisons
- Precision-Recall tradeoff curves
- Feature importance rankings

---

## Acknowledgments

This project builds upon the work of:

1. **Dataset Providers**:  
   [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
   (European cardholders, 2013 transactions)

2. **Machine Learning Libraries**:  
   - Scikit-learn development team  
   - XGBoost/LightGBM/CatBoost contributors  
   - Imbalanced-learn maintainers  

3. **Explainability Tools**:  
   SHAP (Scott Lundberg, Microsoft Research)

4. **Visualization Tools**:  
   Matplotlib, Seaborn, and Plotly communities

---

## License

MIT License - For academic and research purposes only. Not certified for production financial systems.
## 📌 Acknowledgments

- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Open-source libraries:
  - [scikit-learn](https://scikit-learn.org/)
  - [imbalanced-learn](https://imbalanced-learn.org/)
  - [XGBoost](https://xgboost.readthedocs.io/)
  - [LightGBM](https://lightgbm.readthedocs.io/)
  - [CatBoost](https://catboost.ai/)
- [SHAP](https://github.com/shap/shap) by Scott Lundberg
