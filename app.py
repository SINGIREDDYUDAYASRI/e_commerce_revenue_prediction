import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model
model = pickle.load(open('knn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("Marketing Revenue Prediction")

# Inputs
ad_spend = st.number_input("Ad Spend")
price = st.number_input("Price")
discount_rate = st.number_input("Discount Rate")
market_reach = st.number_input("Market Reach")
impressions = st.number_input("Impressions")
ctr = st.number_input("Click Through Rate")
competition = st.number_input("Competition Index")
seasonality = st.number_input("Seasonality Index")
campaign_days = st.number_input("Campaign Duration Days")
clv = st.number_input("Customer Lifetime Value")

channel = st.selectbox("Channel", ["Search", "Social Media", "Email", "Affiliate", "TV", "Influencer"])

if st.button("Predict"):
    input_dict = {
        'ad_spend': ad_spend,
        'price': price,
        'discount_rate': discount_rate,
        'market_reach': market_reach,
        'impressions': impressions,
        'click_through_rate': ctr,
        'competition_index': competition,
        'seasonality_index': seasonality,
        'campaign_duration_days': campaign_days,
        'customer_lifetime_value': clv,
        'channel': channel
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Revenue: {prediction[0]:.2f}")