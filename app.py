import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pickle

# Load pre-trained models and scaler
with open('mlr_model.pkl', 'rb') as file:
    mlr_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('ridge_tuned_model.pkl', 'rb') as file:
    tuned_ridge_model = pickle.load(file)

# Load the dataset
data = pd.read_csv('vehicle_data.csv')  # Replace with your actual dataset

# Preprocessing (same as before)
label_encoders = {}
categorical_columns = ['Vehicle Class']
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
            'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
            'Vehicle Class']
target = 'CO2 Emissions(g/km)'

X = data[features]
y = data[target]
X_scaled = scaler.transform(X)  # Apply the same scaling to new data

# Sidebar: User Inputs (styled)
st.sidebar.title("🚗 CO2 Emissions Prediction")
st.sidebar.markdown("Enter vehicle specifications to predict CO2 emissions (g/km).")

# Input controls for user
engine_size = st.sidebar.slider("Engine Size (L)", min_value=1.0, max_value=6.0, step=0.1)
cylinders = st.sidebar.slider("Number of Cylinders", min_value=3, max_value=16, step=1)
fuel_city = st.sidebar.number_input("Fuel Consumption (City) (L/100 km)", min_value=0.0, step=0.1)
fuel_hwy = st.sidebar.number_input("Fuel Consumption (Highway) (L/100 km)", min_value=0.0, step=0.1)
fuel_comb = st.sidebar.number_input("Fuel Consumption (Combined) (L/100 km)", min_value=0.0, step=0.1)
vehicle_type = st.sidebar.selectbox("Vehicle Class", options=[0, 1])  # Match encoded categories

# Prepare input data for prediction
input_data = np.array([[engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb, vehicle_type]])
input_data_scaled = scaler.transform(input_data)

# Prediction with MLR model
prediction_mlr = mlr_model.predict(input_data_scaled)[0]

# Title and result display
st.title("🌍 CO2 Emissions Predictor")
st.markdown("This model predicts CO2 emissions based on the specifications of a vehicle.")

st.subheader("Predicted CO2 Emissions")
st.metric(label="CO2 Emissions (g/km)", value=f"{prediction_mlr:.2f}")

# Visualization of Engine Size vs CO2 Emissions with Regression Line
st.subheader("Engine Size vs CO2 Emissions")

# Scatter plot of sample data
plt.figure(figsize=(8, 6))
plt.scatter(data['Engine Size(L)'], data['CO2 Emissions(g/km)'], color='blue', label="Data")
plt.plot(data['Engine Size(L)'], mlr_model.predict(X_scaled), color='red', label="Regression Line")

# Labels and title
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Engine Size vs CO2 Emissions")
plt.legend()

# Display the plot
st.pyplot(plt)

# Cross-validation result
cross_val_results = cross_val_score(LinearRegression(), X_scaled, y, cv=5, scoring='neg_mean_squared_error')
st.subheader("Cross-Validation Performance")
st.write(f"Cross-Validation Mean RMSE: {np.sqrt(-cross_val_results.mean()):.2f}")

# Hyperparameter Tuning with Ridge Regression
param_grid = {'alpha': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_scaled, y)

best_ridge_model = grid_search.best_estimator_
st.subheader("Optimized Ridge Regression Model")
st.write(f"Best Ridge Model Hyperparameters: {grid_search.best_params_}")

# Make predictions with the tuned Ridge model
tuned_prediction = best_ridge_model.predict(input_data_scaled)[0]
st.write(f"Tuned Ridge Model Prediction: {tuned_prediction:.2f} g/km")

# Evaluate the tuned Ridge model
y_test_pred_tuned = best_ridge_model.predict(X_scaled)
tuned_test_rmse = mean_squared_error(y, y_test_pred_tuned, squared=False)
tuned_test_r2 = r2_score(y, y_test_pred_tuned)

st.write(f"Tuned Model Testing RMSE: {tuned_test_rmse:.2f}, R²: {tuned_test_r2:.2f}")

# Commercial Section: Highlight benefits
st.markdown("""
---
### Why This Matters?
- **Reduce Environmental Impact**: Understanding the CO2 emissions of vehicles helps reduce the carbon footprint and promote eco-friendly practices.
- **Help Consumers**: Make informed decisions when choosing vehicles based on their environmental impact.
- **Business Advantage**: Ideal for car manufacturers, car buyers, and environmental researchers to assess the carbon emissions of different vehicles.
""")

# Footer with links
st.markdown("""
---
**Developed by [Ashish Patel](https://github.com/YourGitHub)**  
Powered by **Streamlit**, **Scikit-learn**, and **Python**.
""")
