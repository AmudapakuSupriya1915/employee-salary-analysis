import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
from io import BytesIO
import base64

# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Salary Predictor", layout="centered", page_icon="💼")

# Default background image (encoded directly in the script)
def set_default_background():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """, unsafe_allow_html=True)

# Set default background
set_default_background()

# Custom CSS for dark theme with teal accents
st.markdown("""
    <style>
    :root {
        --primary-color: #00CED1;  /* Teal */
        --primary-dark: #008B8B;
        --primary-light: #AFEEEE;
        --text-color: #00CED1;
        --bg-color: rgba(0, 0, 20, 0.85);
        --card-bg: rgba(10, 25, 47, 0.9);
        --border-color: #008B8B;
    }
    
    body {
        color: var(--text-color);
    }
    
    .main-container {
        max-width: 800px;
        margin: 2rem auto;
        padding: 2rem;
        border-radius: 12px;
        background: var(--card-bg);
        box-shadow: 0 4px 30px rgba(0, 206, 209, 0.2);
        backdrop-filter: blur(5px);
        border: 1px solid var(--border-color);
    }
    
    .header {
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .header h1 {
        color: var(--primary-color);
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .header p {
        color: var(--primary-light);
        font-size: 1.1rem;
    }
    
    .form-container {
        padding: 1.5rem;
        background: var(--bg-color);
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
    }
    
    .result-card {
        padding: 1.5rem;
        background: var(--bg-color);
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
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
        background: radial-gradient(circle, rgba(0,206,209,0.8) 0%, rgba(0,206,209,0) 70%);
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
    
    /* Input fields styling */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: rgba(0, 0, 30, 0.7) !important;
        color: var(--primary-light) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Slider styling */
    .stSlider .thumb {
        background-color: var(--primary-color) !important;
    }
    
    .stSlider .track {
        background-color: var(--primary-dark) !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--primary-dark);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 6px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 206, 209, 0.4);
        color: #000 !important;
    }
    
    /* Download button */
    .download-btn {
        background-color: var(--primary-color) !important;
        color: #000 !important;
    }
    
    .download-btn:hover {
        background-color: var(--primary-light) !important;
    }
    
    /* Success message */
    .stAlert {
        background-color: var(--card-bg) !important;
        border-color: var(--primary-color) !important;
        color: var(--primary-light) !important;
    }
    
    /* Hide sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Change all text colors to teal */
    p, div, span, label, h1, h2, h3, h4, h5, h6 {
        color: var(--text-color) !important;
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
            <div style="background: rgba(0, 139, 139, 0.2); padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid var(--border-color);">
                <h4 style="margin: 0; color: var(--primary-light);">Estimated Annual Salary</h4>
                <h2 style="margin: 0.5rem 0; color: var(--primary-color);">₹{prediction:,.2f}</h2>
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