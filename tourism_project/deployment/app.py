import streamlit as st
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="Shivam174/tourism-prediction-model",
                                 filename="best_tourism_model.joblib")
    model = joblib.load(model_path)
    return model

model = load_model()

st.title("Wellness Tourism Package Purchase Predictor")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])

# Add other inputs similarly as appropriate

if st.button("Predict"):
    inputs = pd.DataFrame({'Age': [age], 'Gender': [gender]})
    prediction = model.predict(inputs)
    st.write("Prediction (1 means likely to purchase):", prediction[0])

