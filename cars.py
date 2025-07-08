import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load CSV from C drive
df = pd.read_csv("C:/Users/Rajiv Chaurasiya/Downloads/electric_vehicles_spec_2025.csv.csv")

# Show columns to help you choose the right ones
print(df.columns)

# Suppose you choose to predict Range_km using Battery_kWh
X = df[['battery_capacity_kWh']]
y = df['range_km']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R² Score:", r2_score(y_test, y_pred))
print("Again Just Chill")

# Visualize
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel("Battery (kWh)")
plt.ylabel("Range (km)")
plt.title("Battery vs Range")
plt.legend()
plt.show()
