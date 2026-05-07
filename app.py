import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Configuration
st.set_page_config(
    page_title="FinTech Loan Predictor",
    page_icon="🏦",
    layout="centered"
)

# 2. Custom CSS for Background & Styling
st.markdown("""
<style>
    /* App Background: Professional Gradient */
    .stApp {
        background: linear-gradient(135deg, #E0F7FA 0%, #FFFFFF 50%, #E1F5FE 100%);
    }

    /* Title Styling */
    h1 {
        color: #0D47A1;
        font-weight: 800;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    h3 {
        color: #1565C0;
    }

    /* Input Labels */
    label {
        color: #1A237E;
        font-weight: bold;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #1976D2 0%, #0D47A1 100%);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        box-shadow: 0 4px 15px rgba(13, 71, 161, 0.3);
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #2196F3 0%, #1565C0 100%);
    }

    /* Result Cards */
    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        animation: fadeIn 0.6s ease-in-out;
    }
    .approved {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 2px solid #81C784;
    }
    .rejected {
        background-color: #FFEBEE;
        color: #C62828;
        border: 2px solid #E57373;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #546E7A;
        font-size: 14px;
    }

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# 3. Load Model & Artifacts
@st.cache_resource
def load_model():
    if not os.path.exists('model_files/loan_model.pkl'):
        st.error("❌ Model file not found! Ensure you ran the 'Save Model' code in the notebook.")
        return None, None, None

    model = joblib.load('model_files/loan_model.pkl')
    scaler = joblib.load('model_files/scaler.pkl')
    features = joblib.load('model_files/features.pkl')
    return model, scaler, features


model, scaler, features = load_model()

# 4. UI Layout
if model:
    st.title("🏦 Automated Loan Approval System")
    st.markdown("### <center>AI-Powered Credit Risk Assessment</center>", unsafe_allow_html=True)

    # Center column for the form
    col1, col2, col3 = st.columns([1, 2.5, 1])

    with col2:
        with st.container(border=True):  # Adds a subtle border box
            st.header("📝 Applicant Details")

            with st.form("prediction_form"):
                col_a, col_b = st.columns(2)

                with col_a:
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    married = st.selectbox("Married", ["No", "Yes"])
                    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
                    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

                with col_b:
                    applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=500)
                    coapplicant_income = st.number_input("Co-Applicant Income (₹)", min_value=0, value=0, step=500)
                    loan_amount = st.number_input("Loan Amount (₹ Thousands)", min_value=9, value=100, step=5)
                    loan_term = st.number_input("Loan Term (Months)", min_value=12, value=360, step=12)
                    credit_history = st.selectbox("Credit History (1.0=Good, 0.0=Bad)", [1.0, 0.0])
                    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

                # Submit Button
                submitted = st.form_submit_button("🚀 Predict Loan Status")

                if submitted:
                    # Data Processing
                    input_data = pd.DataFrame({
                        'Gender': [gender], 'Married': [married], 'Dependents': [dependents],
                        'Education': [education], 'Self_Employed': [self_employed],
                        'ApplicantIncome': [applicant_income], 'CoapplicantIncome': [coapplicant_income],
                        'LoanAmount': [loan_amount], 'Loan_Amount_Term': [loan_term],
                        'Credit_History': [credit_history], 'Property_Area': [property_area]
                    })

                    # Clean 'Dependents'
                    input_data['Dependents'] = input_data['Dependents'].replace('3+', '3').astype(float)

                    # Feature Engineering
                    input_data['TotalIncome'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
                    input_data['LoanToIncomeRatio'] = input_data['LoanAmount'] / (input_data['TotalIncome'] + 1e-5)

                    # Encode & Align
                    df_encoded = pd.get_dummies(input_data, columns=['Gender', 'Married', 'Education', 'Self_Employed',
                                                                     'Property_Area'], drop_first=True)

                    # Ensure all training columns exist
                    for col in features:
                        if col not in df_encoded.columns:
                            df_encoded[col] = 0
                    df_encoded = df_encoded[features]

                    # Predict
                    X_scaled = scaler.transform(df_encoded)
                    prediction = model.predict(X_scaled)[0]

                    # Display Result
                    if prediction == 1:
                        st.markdown('<div class="result-box approved">✅ LOAN APPROVED</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-box rejected">❌ LOAN REJECTED</div>', unsafe_allow_html=True)

    st.markdown("<div class='footer'>Powered by NAVTTC AI/ML Project | Student Submission</div>",
                unsafe_allow_html=True)