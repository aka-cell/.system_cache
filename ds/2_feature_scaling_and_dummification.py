import pandas as pd
from sklearn.preprocessing import (  # pip install scikit-learn
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
)

# Create a sample dataset
data = pd.DataFrame(
    {
        "Age": [25, 30, 35, 40, 28, 32, 38, 45],
        "Income": [50000, 60000, 70000, 80000, 55000, 65000, 75000, 90000],
        "Education": [
            "Bachelor",
            "Master",
            "PhD",
            "Master",
            "Bachelor",
            "Master",
            "PhD",
            "Master",
        ],
        "Marital_Status": [
            "Single",
            "Married",
            "Single",
            "Married",
            "Single",
            "Married",
            "Single",
            "Married",
        ],
    }
)
print("\nOriginal Dataset:\n", data)

# Apply Standardization (Z-Score Scaling)
data[["Age_scaled", "Income_scaled"]] = StandardScaler().fit_transform(
    data[["Age", "Income"]]
)
print(
    "\nData after Standardization:\n",
    data[["Age", "Income", "Age_scaled", "Income_scaled"]],
)

# Apply Normalization (Min-Max Scaling)
data[["Age_normalized", "Income_normalized"]] = MinMaxScaler().fit_transform(
    data[["Age", "Income"]]
)
print(
    "\nData after Normalization:\n",
    data[["Age", "Income", "Age_normalized", "Income_normalized"]],
)

# Label Encoding (simple integer encoding)
le1 = LabelEncoder()
le2 = LabelEncoder()

data["Education_code"] = le1.fit_transform(data["Education"])
data["Marital_Status_code"] = le2.fit_transform(data["Marital_Status"])
print("\nData with Encoded Categorical Variables:\n", data)


# Perform Feature Dummification for Categorical Variables
data_dummified = pd.get_dummies(
    data, columns=["Education", "Marital_Status"], drop_first=True
)
print("\nData after Dummification:\n", data_dummified)
