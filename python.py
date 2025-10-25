# 🌾 AI for Sustainable Development
# Project: Predicting Crop Yields for Sustainable Agriculture (SDG 2: Zero Hunger)
# Author: Prince
# Platform: Google Colab / GitHub

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 2️⃣ Load Dataset
# You can replace this with your own dataset link or Kaggle dataset path.
# Example dataset should have columns like: rainfall, temperature, soil_ph, fertilizer, area, yield
url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/crop_yield.csv"
data = pd.read_csv(url)

# Display first few rows
print("Dataset Preview:")
print(data.head())

# 3️⃣ Data Preprocessing
# Check for missing values
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# Fill missing values with column means
data.fillna(data.mean(), inplace=True)

# Define Features (X) and Target (y)
X = data[['rainfall', 'temperature', 'soil_ph', 'fertilizer', 'area']]
y = data['yield']

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Build and Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# 6️⃣ Visualization
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color='green')
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.grid(True)
plt.show()

# 7️⃣ Example Prediction
example = pd.DataFrame({
    'rainfall': [200],
    'temperature': [27],
    'soil_ph': [6.5],
    'fertilizer': [150],
    'area': [2]
})
predicted_yield = model.predict(example)[0]
print(f"\n🌾 Predicted Crop Yield for Sample Input: {predicted_yield:.2f} tons/hectare")

# ✅ SDG Alignment:
# - SDG 2: Zero Hunger
# - Promotes food security and sustainable agriculture

