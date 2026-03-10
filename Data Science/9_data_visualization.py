import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")
# Load Dataset (Titanic dataset for example)
df = sns.load_dataset("titanic")
# Display first few rows
print(df.head())
# Check missing values
print(df.isnull().sum())
# Summary statistics
print(df.describe())
# Count of passengers by class
plt.figure(figsize=(8, 5))
sns.countplot(x="class", data=df, hue="class", palette="Set2", legend=False)
plt.title("Number of Passengers by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()
# Survival count by class

plt.figure(figsize=(8, 5))
sns.barplot(x="class", y="survived", data=df, palette="coolwarm")
plt.title("Survival Rate by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="sex", y="survived", data=df, palette="viridis")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(
    df[df["survived"] == 1]["age"], bins=30, kde=True, color="green", label="Survived"
)
sns.histplot(
    df[df["survived"] == 0]["age"], bins=30, kde=True, color="red", label="Not Survived"
)
plt.legend()
plt.title("Age Distribution of Titanic Survivors vs Non-Survivors")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
