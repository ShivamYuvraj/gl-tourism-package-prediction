%%writefile /content/tourism_project/deployment/app.py
import streamlit as st
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd

st.set_page_config(page_title="Wellness Tourism Package Purchase Predictor", layout="wide")

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="Shivam174/tourism-prediction-model",
            filename="best_tourism_model.joblib"
        )
    except Exception as e:
        # If hf_hub_download fails (private model / missing token), propagate so Streamlit shows an error
        raise RuntimeError(
            "Could not download model from Hugging Face. "
            "If the model repo is private, make sure HF_TOKEN is available in the environment."
        ) from e
    model = joblib.load(model_path)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

st.title("Wellness Tourism Package Purchase Predictor")

# --- Defaults chosen from training-data medians / sensible ranges ---
# Numerical defaults are chosen to match your training data medians (approx.)
# Categorical options are taken from the training data.

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=36, step=1)
    TypeofContact = st.selectbox("Type of Contact", options=["Company Invited", "Self Enquiry"])
    CityTier = st.selectbox("City Tier", options=[1, 2, 3], index=0)
    Occupation = st.selectbox("Occupation", options=["Salaried", "Free Lancer", "Small Business", "Large Business"])
    Gender = st.selectbox("Gender", options=["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", options=["Married", "Single", "Unmarried", "Divorced"])

with col2:
    Designation = st.selectbox("Designation", options=["AVP", "Executive", "Manager", "Senior Manager", "VP"])
    ProductPitched = st.selectbox("Product Pitched", options=["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=10, value=3, step=1)
    PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3, step=1)
    NumberOfTrips = st.number_input("Number Of Trips", min_value=0, max_value=100, value=3, step=1)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, max_value=20, value=4, step=1)
    DurationOfPitch = st.number_input("Duration Of Pitch (seconds)", min_value=1, max_value=1000, value=14, step=1)
    MonthlyIncome = st.number_input("Monthly Income (monthly)", min_value=0.0, max_value=1_000_000.0, value=22418.0, step=100.0, format="%.2f")

# When user clicks Predict
if st.button("Predict"):
    # Clean / normalize some inputs so they align with how the model was trained
    gender_clean = str(Gender).strip()
    # training data contained both "Female" and a bad "Fe Male" entry; normalize to "Female"
    if gender_clean.lower().replace(" ", "") in ("femal","female","femalE","fem"):
        gender_clean = "Female"
    elif gender_clean.lower().replace(" ", "") in ("male",):
        gender_clean = "Male"

    # Build input dataframe with EXACT column names your pipeline expects
    input_df = pd.DataFrame([{
        'Age': Age,
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'PreferredPropertyStar': PreferredPropertyStar,
        'NumberOfTrips': NumberOfTrips,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'NumberOfFollowups': NumberOfFollowups,
        'DurationOfPitch': DurationOfPitch,
        'MonthlyIncome': MonthlyIncome,
        'TypeofContact': TypeofContact,
        'CityTier': CityTier,
        'Occupation': Occupation,
        'Gender': gender_clean,
        'MaritalStatus': MaritalStatus,
        'Designation': Designation,
        'ProductPitched': ProductPitched
    }])

    st.subheader("Inputs")
    st.write(input_df.T)  # transpose so it looks nice in the UI

    try:
        # Predict class
        pred = model.predict(input_df)
        # Predict probability if available
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[:, 1][0]

        st.success(f"Prediction: **{int(pred[0])}**  —  (1 = likely to purchase, 0 = unlikely)")
        if proba is not None:
            st.info(f"Model probability of purchase: **{proba:.2%}**")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
