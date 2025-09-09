# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load cleaned dataset
df = pd.read_csv("train_cleaned.csv")

# 2. Basic overview
print("🔹 Dataset Shape:", df.shape)
print("\n🔹 Summary statistics for numerical features:\n", df.describe())

# 3. Target variable distribution
plt.figure(figsize=(8,5))
sns.histplot(df['SalePrice'], bins=50, kde=True)
plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Count")
plt.show()

# 4. Correlation heatmap (numerical features only)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr = df[numerical_cols].corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features Only)")
plt.show()

# 5. Scatter plots for top correlated numerical features
top_corr_features = corr['SalePrice'].abs().sort_values(ascending=False)[1:6].index
print("\n🔹 Top 5 features correlated with SalePrice:", top_corr_features.tolist())

for feature in top_corr_features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[feature], y=df['SalePrice'])
    plt.title(f"SalePrice vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.show()

# 6. Categorical features analysis (average SalePrice per category)
categorical_features = df.select_dtypes(include=['object']).columns

# Example categorical features to plot
for feature in ['Neighborhood', 'HouseStyle', 'BldgType', 'OverallQual']:
    if feature in categorical_features:
        plt.figure(figsize=(10,5))
        avg_price = df.groupby(feature)['SalePrice'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_price.index, y=avg_price.values)
        plt.xticks(rotation=45)
        plt.ylabel("Average SalePrice")
        plt.title(f"Average SalePrice by {feature}")
        plt.show()
