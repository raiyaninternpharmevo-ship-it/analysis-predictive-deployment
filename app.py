import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

# -------------------------------
# Google Drive Model Download
# -------------------------------
MODEL_FILE = "ML-MODELLING.pkl"
FILE_ID = "12AnL6fTpJHOa_nz2DQh4CnkERX0-UPfI"

def download_model_from_drive():
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    response = requests.get(url)
    with open(MODEL_FILE, "wb") as f:
        f.write(response.content)

if not os.path.exists(MODEL_FILE):
    st.info("⬇️ Downloading ML model from Google Drive...")
    download_model_from_drive()

model = joblib.load(MODEL_FILE)

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Pharma Sales Prediction", layout="centered")

st.title("💊 Pharma Sales Prediction App")
st.write("Predict Net Sales Value using Machine Learning Model")

st.divider()

# User Inputs
units = st.number_input("Units Sold", min_value=0, value=100)
bonus = st.number_input("Bonus Units", min_value=0, value=0)
discount = st.number_input("Discount (%)", min_value=0.0, value=5.0)
tprice = st.number_input("Total Price", min_value=0.0, value=1000.0)

# Prediction
if st.button("Predict Sales Value"):
    input_data = pd.DataFrame({
        "Units": [units],
        "Bonus": [bonus],
        "Discount": [discount],
        "TPrice": [tprice]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Net Sales Value: ₹ {prediction:,.2f}")

st.divider()
st.caption("🚀 Deployed using Streamlit | Model loaded from Google Drive")
