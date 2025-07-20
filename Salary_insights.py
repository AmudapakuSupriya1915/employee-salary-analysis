import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
from io import BytesIO
import base64

# Remove sidebar
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Set page configuration
st.set_page_config(page_title="Salary Predictor", layout="centered", page_icon="💼")

# Custom CSS for professional UI
st.markdown("""
    <style>
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 12px;
        background: white;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
    }
    
    .header {
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .header h1 {
        color: #2c3e50;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        color: #7f8c8d;
        font-size: 1rem;
    }
    
    .form-container {
        padding: 1.5rem;
        background: #f9f9f9;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    
    .result-card {
        padding: 1.5rem;
        background: #f0f7ff;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin-top: 1.5rem;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    .sparkle {
        position: relative;
        overflow: hidden;
    }
    
    .sparkle::after {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0) 70%);
        transform: scale(0);
        opacity: 0;
        pointer-events: none;
    }
    
    .sparkle.active::after {
        animation: sparkle 0.6s ease-out;
    }
    
    @keyframes sparkle {
        0% { transform: scale(0); opacity: 1; }
        100% { transform: scale(1); opacity: 0; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 6px;
        font-size: 1rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .download-btn {
        background-color: #27ae60 !important;
    }
    
    .download-btn:hover {
        background-color: #219653 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained model
try:
    model = joblib.load("linearmodel.pkl")
except FileNotFoundError:
    st.error("Model file not found. Make sure 'linearmodel.pkl' exists.")
    st.stop()

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header section
st.markdown("""
    <div class="header">
        <h1>Salary Prediction Tool</h1>
        <p>Get accurate salary estimates based on your experience and role</p>
    </div>
""", unsafe_allow_html=True)

# Form container
with st.form("prediction_form"):
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        years = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
        bonus = st.number_input("Bonus Percentage", min_value=0.0, max_value=100.0, step=0.5, value=10.0)
    with col2:
        jobrate = st.slider("Job Performance Rating", 1.0, 5.0, 3.5, 0.5)
        department = st.selectbox("Department", ["Engineering", "Sales", "HR", "Finance", "Operations"])
    
    role = st.selectbox("Position Level", ["Junior", "Mid", "Senior", "Lead", "Manager"])
    email = st.text_input("Email (optional, for report delivery)")
    
    submit_button = st.form_submit_button("Predict Salary")
    st.markdown('</div>', unsafe_allow_html=True)

# On Submit
if submit_button:
    # Add sparkle effect
    st.markdown("""
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const button = document.querySelector('.stButton button');
            button.addEventListener('click', function() {
                this.classList.add('active');
                setTimeout(() => this.classList.remove('active'), 600);
            });
        });
        </script>
    """, unsafe_allow_html=True)
    
    with st.spinner("Analyzing your data..."):
        time.sleep(1.5)
    
    # Calculate prediction
    jobrate_adj = jobrate + (bonus / 100) + (2 if role in ['Senior', 'Lead', 'Manager'] else 0)
    X = np.array([[years, jobrate_adj]])
    prediction = model.predict(X)[0]
    
    # Display results
    st.markdown(f"""
        <div class="result-card">
            <h3>Salary Prediction Result</h3>
            <div style="margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>Years of Experience:</span>
                    <span><strong>{years}</strong></span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>Performance Rating:</span>
                    <span><strong>{jobrate}/5.0</strong></span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>Bonus Percentage:</span>
                    <span><strong>{bonus}%</strong></span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>Position Level:</span>
                    <span><strong>{role}</strong></span>
                </div>
            </div>
            <div style="background: #e1f0ff; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="margin: 0; color: #2c3e50;">Estimated Annual Salary</h4>
                <h2 style="margin: 0.5rem 0; color: #3498db;">₹{prediction:,.2f}</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # CSV Export
    df = pd.DataFrame([{
        "Years of Experience": years,
        "Performance Rating": jobrate,
        "Bonus Percentage": bonus,
        "Department": department,
        "Position Level": role,
        "Predicted Salary": round(prediction, 2)
    }])
    
    csv_io = BytesIO()
    df.to_csv(csv_io, index=False)
    
    st.download_button(
        label="Download Report as CSV",
        data=csv_io.getvalue(),
        file_name="salary_prediction_report.csv",
        mime="text/csv",
        key="download-csv"
    )
    
    if email:
        st.success(f"Report will be sent to: {email}")

st.markdown('</div>', unsafe_allow_html=True)