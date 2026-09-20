import os
import streamlit as st
import pandas as pd
import skops.io as sio
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="SuperKart Sales Predictor", page_icon="🛒", layout="wide")
st.title("🛒 SuperKart Demand Prediction Service")
st.markdown("Enter product and store attributes below to estimate total store sales.")

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "HSSHETTY01/SuperKart-RandomForest-Model")

@st.cache_resource
def load_model():
    if os.path.exists("model.skops"):
        model_path = "model.skops"
    elif os.path.exists("artifacts/model.skops"):
        model_path = "artifacts/model.skops"
    else:
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="model.skops")
    return sio.load(model_path, trusted=sio.get_untrusted_types(file=model_path))

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5)
        sugar_content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        allocated_area = st.number_input("Allocated Display Area Ratio", min_value=0.0, max_value=1.0, value=0.06)
        product_type = st.selectbox(
            "Product Type",
            [
                "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
                "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene",
                "Hard Drinks", "Canned", "Breads", "Starchy Foods", "Others", "Seafood"
            ]
        )
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=140.0)

    with col2:
        store_year = st.number_input("Store Establishment Year", min_value=1950, max_value=2026, value=2008)
        store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
        city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox(
            "Store Type",
            ["Departmental Store", "Supermarket Type 1", "Supermarket Type 2", "Food Mart"]
        )

    submit_btn = st.form_submit_button("Predict Sales")

if submit_btn:
    input_df = pd.DataFrame([{
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Establishment_Year": int(store_year),
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
    }])

    prediction = model.predict(input_df)[0]
    st.subheader(f"Predicted Total Sales: ₹ {prediction:,.2f}")
