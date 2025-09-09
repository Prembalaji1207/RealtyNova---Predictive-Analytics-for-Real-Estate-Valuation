# model_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from math import sqrt
import os

# 1. Load cleaned dataset
df = pd.read_csv("train_cleaned.csv")

# 2. Feature selection
# Top correlated numerical features
numerical_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']

# Simple list of categorical features
categorical_features = ['Neighborhood', 'HouseStyle', 'BldgType']

# 3. Define target and features
X = df[numerical_features + categorical_features]
y = df['SalePrice']

# 4. One-hot encoding for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # keep numerical features as-is
)

# 5. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# 7. Ensure models folder exists
os.makedirs("models", exist_ok=True)

# 8. Train & evaluate
for name, model in models.items():
    # Create pipeline: preprocessing + model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluate
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")

    # Save the trained model
    joblib.dump(pipeline, f"models/{name.replace(' ', '_')}_model.pkl")
