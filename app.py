
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
st.markdown("""
    <style>
    .toggle-switch {
        display: flex;
        align-items: center;
        gap: 0.5em;
    }
    .toggle-switch label {
        font-weight: bold;
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit toggle for dark/light mode
dark_mode = st.sidebar.checkbox("", value=False, key="dark_mode_toggle")
mode_text = "Dark Mode" if dark_mode else "Light Mode"
st.sidebar.markdown(f"<span style='font-weight:bold;font-size:1.1em'>{mode_text}</span>", unsafe_allow_html=True)

# Apply theme based on toggle
if dark_mode:
    sns.set_style("darkgrid")
    plt.style.use('dark_background')
    st.markdown(
        "<style>body { background-color: #181818; color: #FAFAFA; }</style>",
        unsafe_allow_html=True
    )
else:
    sns.set_style("whitegrid")
    plt.style.use('default')
    st.markdown(
        "<style>body { background-color: #FAFAFA; color: #181818; }</style>",
        unsafe_allow_html=True
    )

st.title("Customer Churn Prediction Dashboard")


# XGBoost feature importance
xgb_features = [
    "AvgMonthlySpend", "Contract", "MonthlyCharges", "AvgChargesPerMonth",
    "PaymentMethod", "InternetService", "OnlineSecurity"
]
xgb_importance = [0.153986, 0.092029, 0.086836, 0.073671, 0.030797, 0.013527, 0.005072]

# Random Forest feature importance
rf_features = [
    "Contract", "AvgMonthlySpend", "MonthlyCharges", "AvgChargesPerMonth",
    "InternetService", "PaymentMethod", "OnlineSecurity"
]
rf_importance = [0.121618, 0.068478, 0.050483, 0.032729, 0.037560, 0.024517, 0.014493]

col5, col6 = st.columns(2)
with col5:
    st.markdown("**XGBoost Feature Importance**")
    fig_xgb_feat, ax_xgb_feat = plt.subplots()
    ax_xgb_feat.pie(
        xgb_importance,
        labels=xgb_features,
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.dark_palette("#a63c29", len(xgb_features))
    )
    ax_xgb_feat.axis('equal')
    st.pyplot(fig_xgb_feat)

with col6:
    st.markdown("**Random Forest Feature Importance**")
    fig_rf_feat, ax_rf_feat = plt.subplots()
    ax_rf_feat.pie(
        rf_importance,
        labels=rf_features,
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.dark_palette("#69d", len(rf_features))
    )
    ax_rf_feat.axis('equal')
    st.pyplot(fig_rf_feat)
st.sidebar.header("Upload Customer Data CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
# plot aoc comparison as grouped pie charts
xgb_aoc=0.92
rf_aoc=0.96
col7,col8= st.columns(2)
with col7:
    st.markdown("**XGBoost AUC**")
    fig_xgb_feat, ax_xgb_feat = plt.subplots(figsize=(5, 4))
    ax_xgb_feat.pie(
        [xgb_aoc, 1 - xgb_aoc],
        labels=['AUC: {:.2f}'.format(xgb_aoc), 'Rest'],
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette("flare", 2)
    )
    ax_xgb_feat.axis('equal')
    st.pyplot(fig_xgb_feat)

with col8:
    st.markdown("**Random Forest AUC**")
    fig_rf_feat, ax_rf_feat = plt.subplots(figsize=(5, 4))
    ax_rf_feat.pie(
        [rf_aoc, 1 - rf_aoc],
        labels=['AUC: {:.2f}'.format(rf_aoc), 'Rest'],
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette("crest", 2)
    )
    ax_xgb_feat.axis('equal')
    st.pyplot(fig_rf_feat)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    if 'customerID' in df.columns:
        customer_ids = df['customerID']
        df_model_input = df.drop(columns=['customerID'])
    else:
        st.error("❌ 'customerID' column is required in your file for prediction tracking.")
        st.stop()

    st.subheader("Input Data Overview")
    st.dataframe(df.head())

    # Load pre-trained model (assumes model is saved as 'best_xgb_churn_model.pkl')
    try:
        model1 = joblib.load('best_churn_model.pkl')
    except FileNotFoundError:
        st.error("❌ Model file 'xgb_model.pkl' not found. Please place it in the same folder as this script.")
    try:
        model2= joblib.load('best_xgb_churn_model.pkl')
    except FileNotFoundError:
        st.error("❌ Model file 'best_xgb_churn_model.pkl' not found. Please place it in the same folder as this script.")
    # Feature engineering and preprocessing would go here
    X_encoded = transform(df_model_input)

    try:
        # Predictions (assuming model works with raw df_model_input)
        probs1 = model1.predict_proba(X_encoded)[:, 1]
        probs2 = model2.predict_proba(X_encoded)[:, 1]
        predictions1 = (probs1 > 0.5).astype(int)
        predictions2 = (probs2 > 0.5).astype(int)
        # Combine predictions with customerID
        output_df = pd.DataFrame({
            'customerID': customer_ids,
            'ChurnProbabilityByRF': probs1,
            'ChurnProbabilityByXGBoost': probs2,
            'ChurnPredictionByRF': predictions1,
            'ChurnPredictionByXGBoost': predictions2
        })
        # Download button
        csv = output_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )



    except FileNotFoundError:
        st.error("❌ Model file 'xgb_model.pkl' not found. Please place it in the same folder as this script.")
    # Grouped Pie Charts: Churn vs Retain Breakdown
    st.subheader("Churn vs Retain Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest**")
        churn_counts_rf = output_df['ChurnPredictionByRF'].value_counts().sort_index()
        fig_rf, ax_rf = plt.subplots(figsize=(5, 4))
        
        
        
        mycolors = ["#550C0C", "#B42424"]
        labels_rf = ["Retain", "Churn"]
        ax_rf.pie(churn_counts_rf, colors=mycolors, labels=labels_rf, autopct='%1.1f%%', startangle=90)
        ax_rf.axis('equal')
        st.pyplot(fig_rf)
    with col2:
        mycolors = ["#0C1355", "#243EB4"]
        st.markdown("**XGBoost**")
        churn_counts_xgb = output_df['ChurnPredictionByXGBoost'].value_counts().sort_index()
        fig_xgb, ax_xgb = plt.subplots(figsize=(5, 4))
        
        ax_xgb.pie(churn_counts_xgb, colors=mycolors, labels=labels_rf, autopct='%1.1f%%', startangle=90)
        ax_xgb.axis('equal')
        st.pyplot(fig_xgb)

    # Grouped Histograms: Churn Probability Distribution
    st.subheader("Churn Probability Distribution")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Random Forest**")
        fig_hist_rf, ax_hist_rf = plt.subplots(figsize=(5, 4))
        sns.histplot(output_df['ChurnProbabilityByRF'], bins=20, kde=True, color="#BE0F02", ax=ax_hist_rf)
        ax_hist_rf.set_xlabel("Churn Probability")
        ax_hist_rf.set_ylabel("Number of Customers")
        st.pyplot(fig_hist_rf)
    with col4:
        st.markdown("**XGBoost**")
        fig_hist_xgb, ax_hist_xgb = plt.subplots(figsize=(5, 4))
        sns.histplot(output_df['ChurnProbabilityByXGBoost'], bins=20, kde=True, color="#4C8BAF", ax=ax_hist_xgb)
        ax_hist_xgb.set_xlabel("Churn Probability")
        ax_hist_xgb.set_ylabel("Number of Customers")
        st.pyplot(fig_hist_xgb)
    # Display total customer count
    st.sidebar.markdown(f"**Total At-Risk Customers:** {len(output_df)}")
    st.sidebar.subheader("Top 10 At-Risk Customers")
    high_risk = output_df[
        (output_df['ChurnProbabilityByRF'] > 0.5) &
        (output_df['ChurnProbabilityByXGBoost'] > 0.5)
    ]
    top_risk = high_risk.sort_values(
        by=['ChurnProbabilityByXGBoost', 'ChurnProbabilityByRF'],
        ascending=False
    ).head(10)
    st.sidebar.write(top_risk[['customerID', 'ChurnProbabilityByXGBoost', 'ChurnProbabilityByRF']])

else:
    st.sidebar.warning("📂 Please upload a CSV file to proceed.")
    
