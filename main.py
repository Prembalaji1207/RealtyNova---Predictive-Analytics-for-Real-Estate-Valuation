# load_and_inspect.py
import pandas as pd

# 1. Load the dataset
df = pd.read_csv("train.csv")

# 2. Basic info
print("🔹 Dataset Shape (rows, columns):", df.shape)
print("\n🔹 First 5 rows of the dataset:\n", df.head())

# 3. Column names
print("\n🔹 Column Names:\n", df.columns.tolist())

# 4. Data types and non-null counts
print("\n🔹 Dataset Info:")
print(df.info())

# 5. Summary statistics (numerical columns only)
print("\n🔹 Summary Statistics:\n", df.describe())

# 6. Missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\n🔹 Missing Values (Top 20):\n", missing.head(20))

# 7. Split categorical & numerical features
categorical = df.select_dtypes(include=['object']).columns
numerical = df.select_dtypes(exclude=['object']).columns

print("\n🔹 Number of Categorical Features:", len(categorical))
print("🔹 Number of Numerical Features:", len(numerical))
print("\n🔹 Sample Categorical Features:", categorical[:10].tolist())
print("🔹 Sample Numerical Features:", numerical[:10].tolist())
