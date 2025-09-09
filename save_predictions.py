import pandas as pd
import joblib
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Load trained model
model = joblib.load("models/XGBoost_model.pkl")

# Load features
df = pd.read_csv("train_cleaned.csv")
numerical_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
categorical_features = ['Neighborhood', 'HouseStyle', 'BldgType']
X = df[numerical_features + categorical_features]

# Predict
df['PredictedPrice'] = model.predict(X)

# Save CSV for Power BI
df[['SalePrice', 'PredictedPrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
    'TotalBsmtSF', 'Neighborhood', 'HouseStyle', 'BldgType']].to_csv('data/predictions.csv', index=False)

print("✅ Predictions saved to data/predictions.csv")
