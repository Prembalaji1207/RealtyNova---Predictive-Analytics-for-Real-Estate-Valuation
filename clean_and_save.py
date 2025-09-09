# clean_and_save.py

import pandas as pd

# 1. Load dataset
df = pd.read_csv("train.csv")

# 2. Drop columns with too many missing values
cols_to_drop = ["PoolQC", "MiscFeature", "Alley", "Fence"]
df = df.drop(columns=cols_to_drop)

# 3. Fill missing values

# LotFrontage: fill with median of each Neighborhood
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)

# Garage-related columns
for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
    df[col] = df[col].fillna("None")

df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
df["GarageCars"] = df["GarageCars"].fillna(0)
df["GarageArea"] = df["GarageArea"].fillna(0)

# Basement-related columns
for col in [
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"
]:
    df[col] = df[col].fillna("None")

for col in ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]:
    df[col] = df[col].fillna(0)

# Masonry veneer
df["MasVnrType"] = df["MasVnrType"].fillna("None")
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

# Electrical (only 1 missing → fill with most common)
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

# 4. Save cleaned dataset
df.to_csv("train_cleaned.csv", index=False)

print("✅ Cleaning done! Saved as data/train_cleaned.csv")
