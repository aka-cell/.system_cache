# =============================================================================
# A. Reading Data from CSV and JSON into a Dataframe
# =============================================================================

import pandas as pd

# -----------------------------------------------------------------------------
# 1: Read Data from a CSV File
# -----------------------------------------------------------------------------
data_csv = {
    "Student_ID": [101, 102, 103, 104, 105],
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Math": [85, 78, 92, 88, 76],
    "Science": [90, 74, 89, 95, 80],
    "English": [88, 69, 94, 91, 72],
}
df_csv = pd.DataFrame(data_csv)
print("Dataset loaded from CSV:")
print(df_csv.head())
df_csv.info()
# -----------------------------------------------------------------------------
# 2: Read Data from a JSON File
# -----------------------------------------------------------------------------
data_json = [
    {"Product_ID": 1, "Product_Name": "Laptop", "Price": 75000, "Stock": 10},
    {"Product_ID": 2, "Product_Name": "Mobile", "Price": 30000, "Stock": 25},
    {"Product_ID": 3, "Product_Name": "Tablet", "Price": 20000, "Stock": 15},
    {"Product_ID": 4, "Product_Name": "Headphones", "Price": 3000, "Stock": 50},
    {"Product_ID": 5, "Product_Name": "Smartwatch", "Price": 15000, "Stock": 20},
]
df_json = pd.DataFrame(data_json)
print("\nDataset loaded from JSON:")
print(df_json.head())
df_json.info()


# =============================================================================
# B. Performing Basic Data Pre-Processing
# =============================================================================

# import pandas as pd
import seaborn as sns

# Load dataset
df = sns.load_dataset("titanic")

print("Dataset loaded:")
print(df.head())

print("\nDataset Information:")
df.info()

# Missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values only in numeric columns
df_filled = df.copy()

numeric_cols = df_filled.select_dtypes(include=["number"]).columns
df_filled[numeric_cols] = df_filled[numeric_cols].fillna(0)

print("\nAfter fillna(0):")
print(df_filled.head(10))

df_filled.to_excel("titanic_replaced.xlsx", index=False)

# Drop missing values
df_dropped = df.dropna()

print("\nAfter dropna():")
print(df_dropped.head(10))

df_dropped.info()

df_dropped.to_excel("titanic_dropped.xlsx", index=False)

# Filtering
survived = df[df["survived"] == 1]
print("\nPassengers who survived:")
print(survived.head())

# Sorting
sorted_fare = df.sort_values(by="fare", ascending=False)
print("\nSorted by Fare:")
print(sorted_fare.head())

# Grouping
grouped_class = df.groupby("pclass")["age"].mean()
print("\nAverage Age per Passenger Class:")
print(grouped_class)
