
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
def transform(data):
    # normalize the numeric columns
    scaler = MinMaxScaler()
    data[['tenure']] = scaler.fit_transform(data[['tenure']])
    data[['MonthlyCharges']] = scaler.fit_transform(data[['MonthlyCharges']])

    # Convert 'TotalCharges' to numeric, coerce errors to NaN
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    # Fill NaN values in 'TotalCharges' with the mean
    data.fillna({'TotalCharges': data['TotalCharges'].mean()}, inplace=True)
    # Now scale
    data[['TotalCharges']] = scaler.fit_transform(data[['TotalCharges']])
    
    # Feature engineering
    data['AvgMonthlySpend'] = data['TotalCharges'] / (data['tenure'] + 1)
    data['AvgChargesPerMonth'] = data['TotalCharges'] / data['tenure'].replace(0, 1)
    data.drop(['tenure', 'TotalCharges'], axis=1, inplace=True)
    # Encoding categorical variables
    
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].astype(str)
    le=LabelEncoder()
    data_encoded = data.copy()
    for col in categorical_columns:
        data_encoded[col] = le.fit_transform(data_encoded[col])
    selected_features = [
    'Contract', 'AvgMonthlySpend', 'MonthlyCharges', 
    'AvgChargesPerMonth', 'InternetService',
    'PaymentMethod', 'OnlineSecurity'
    ]
    # selecting the features for the model
    data_encoded = data_encoded[selected_features]
    return data_encoded

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("🔍 Customer Churn Prediction Dashboard")

st.sidebar.header("Upload Customer Data CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    if 'customerID' in df.columns:
        customer_ids = df['customerID']
        df_model_input = df.drop(columns=['customerID'])
    else:
        st.error("❌ 'customerID' column is required in your file for prediction tracking.")
        st.stop()

    st.subheader("📊 Input Data Overview")
    st.dataframe(df.head())

    # Load pre-trained model (assumes model is saved as 'best_xgb_churn_model.pkl')
    try:
        model = joblib.load('best_churn_model.pkl')
    except FileNotFoundError:
        st.error("❌ Model file 'xgb_model.pkl' not found. Please place it in the same folder as this script.")
    try:
        model= joblib.load('best_xgb_churn_model.pkl')
    except FileNotFoundError:
        st.error("❌ Model file 'best_xgb_churn_model.pkl' not found. Please place it in the same folder as this script.")
    # Feature engineering and preprocessing would go here
    X_encoded = transform(df_model_input)

    try:
        # Predictions (assuming model works with raw df_model_input)
        probs = model.predict_proba(X_encoded)[:, 1]
        predictions = (probs > 0.5).astype(int)

        # Combine predictions with customerID
        output_df = pd.DataFrame({
            'customerID': customer_ids,
            'ChurnProbability': probs,
            'ChurnPrediction': predictions
        })

        st.subheader("📈 Churn Predictions")
        st.write(output_df)

        st.metric("Predicted Churn Rate", f"{(output_df['ChurnPrediction'].mean() * 100):.2f}%")

        # Download button
        csv = output_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.error("❌ Model file 'xgb_model.pkl' not found. Please place it in the same folder as this script.")

else:
    st.sidebar.warning("📂 Please upload a CSV file to proceed.")
