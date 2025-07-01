# Bike-Sharing-Demand-Prediction
**Bike-Sharing Demand Prediction** is a machine learning project that forecasts the number of bikes that will be rented at a given time based on factors like weather, season, temperature, humidity, and day of the week. This helps bike rental companies optimize fleet management and improve user availability.
# 🚴‍♀️ BoomBikes: Bike-Sharing Demand Prediction

This project predicts daily bike rental demand using historical data and machine learning techniques. It helps optimize resource planning and availability for bike-sharing services.

---

## 📊 Problem Statement

The goal is to forecast the **count of total bikes rented (`cnt`)** based on features like temperature, humidity, windspeed, season, month, weekday, holiday, and weather conditions.

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib, Seaborn
- Joblib (for model saving)

---

## 📁 Project Structure

BoomBikes_Prediction/
├── app.py # Streamlit app
├── BoomBikes.csv # Dataset
├── best_random_forest_model.pkl # Saved model
├── scaler.pkl # Saved scaler
├── requirements.txt # Dependencies
├── README.md # This file



---

## 🔍 Key Steps

1. **Data Preprocessing**  
   - One-hot encoding
   - Scaling numerical features
   - Feature selection

2. **Model Training**  
   - Linear Regression
   - Random Forest Regressor
   - Hyperparameter Tuning (GridSearchCV)

3. **Model Evaluation**  
   - R² Score, RMSE, RMSLE
   - Cross-validation

4. **Deployment**  
   - Front-end built using **Streamlit**
   - Predict rentals in real time based on user inputs

---

## 🚀 How to Run

### ▶️ Locally

1. Install dependencies:

```bash
pip install -r requirements.txt


Run the app:
streamlit run app.py
