import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('best_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="BoomBikes Rental Prediction", layout="centered")
st.title("🚴 BoomBikes Rental Count Predictor")
st.markdown("Enter the inputs to predict the number of bike rentals.")

# Numeric Inputs
yr = st.selectbox("Year", [0, 1])  # 0 = 2018, 1 = 2019
holiday = st.selectbox("Holiday?", [0, 1])
workingday = st.selectbox("Working Day?", [0, 1])
temp = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5)
atemp = st.slider("Feels-like Temperature", 0.0, 1.0, 0.5)
hum = st.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.slider("Windspeed", 0.0, 1.0, 0.2)

# Categorical One-hot Inputs
season = st.selectbox("Season", ['fall', 'spring', 'summer', 'winter'])
mnth = st.selectbox("Month", ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
weekday = st.selectbox("Weekday", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
weathersit = st.selectbox("Weather Situation", ['clear', 'mist_cloudy', 'light_snow_heavy_rain'])

# Manual one-hot encoding based on your final model columns
def manual_one_hot(val, options):
    return [1 if val == opt else 0 for opt in options]

# Prepare feature vector in order
final_input = [
    yr, holiday, workingday, temp, atemp, hum, windspeed
]

# season_spring, season_summer, season_winter (fall is base)
final_input += manual_one_hot(season, ['spring', 'summer', 'winter'])

# mnth_Aug to mnth_Sep (Jan is in list, Apr is base)
final_input += manual_one_hot(mnth, [
    'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep'
])

# weekday_Mon to weekday_Wed (Fri is base)
final_input += manual_one_hot(weekday, [
    'Mon', 'Sat', 'Sun', 'Thu', 'Tue', 'Wed'
])

# weathersit_light_snow_heavy_rain, weathersit_mist_cloudy (clear is base)
final_input += manual_one_hot(weathersit, [
    'light_snow_heavy_rain', 'mist_cloudy'
])

# Convert to array and scale
final_input = np.array(final_input).reshape(1, -1)
scaled_input = scaler.transform(final_input)

# Predict
if st.button("Predict Bike Rentals"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"🚲 Predicted Bike Rentals: {int(prediction)}")

