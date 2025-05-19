import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models
rf_model = joblib.load('random_forest_model.pkl')
smote_rf_model = joblib.load('smote_random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
lgbm_model = joblib.load('lightgbm_model.pkl')
cat_model = joblib.load('catboost_model.pkl')
ada_model = joblib.load('adaboost_model.pkl')

#  Title
st.title("💳 Credit Card Fraud Detection")

# Sidebar model selection
model_choice = st.sidebar.selectbox("Select model:", 
    ('RandomForest', 'SMOTE RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'AdaBoost'))

# Transaction input fields
def get_user_input():
    inputs = {}
    for i in range(1, 29):
        inputs[f'V{i}'] = st.sidebar.number_input(f'V{i}', value=0.0)
    inputs['Amount'] = st.sidebar.number_input('Amount', value=0.0)
    return pd.DataFrame([inputs])

input_df = get_user_input()
input_df['Time'] = 0
expected_cols = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
input_df = input_df[expected_cols]

# Prediction button
if st.button("🚀 Predict"):

    model = {
        'RandomForest': rf_model,
        'SMOTE RandomForest': smote_rf_model,
        'XGBoost': xgb_model,
        'LightGBM': lgbm_model,
        'CatBoost': cat_model,
        'AdaBoost': ada_model
    }[model_choice]

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader(f"🔍 Prediction: {'FRAUD ⚠️' if prediction[0]==1 else 'NOT FRAUD ✅'}")
    st.write(f"**Fraud Probability:** `{prediction_proba[0][1]*100:.2f}%`")

    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        'Model': model_choice,
        'Prediction': 'Fraud' if prediction[0]==1 else 'Not Fraud',
        'Fraud Probability (%)': round(prediction_proba[0][1]*100, 2)   
    })

#  Show history
if 'history' in st.session_state and st.session_state.history:
    st.write("### 📊 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))
