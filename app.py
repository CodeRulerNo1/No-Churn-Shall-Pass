
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("🔍 Customer Churn Prediction Dashboard")

st.sidebar.header("Upload Customer Data (No 'Churn' column)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

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
        with open("best_xgb_churn_model.pkl", "rb") as file:
            model = pickle.load(file)

        # Feature engineering and preprocessing would go here
        # Example (uncomment and customize):
        # X_encoded = preprocessor.transform(df_model_input)

        # Predictions (assuming model works with raw df_model_input)
        probs = model.predict_proba(df_model_input)[:, 1]
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
    st.warning("📂 Please upload a CSV file to proceed.")
