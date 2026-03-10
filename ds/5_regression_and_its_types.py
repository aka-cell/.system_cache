import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ================================
# Implement simple linear regression using a dataset.
# ================================


# ================================
# PART 1 : SIMPLE LINEAR REGRESSION
# ================================

np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.randn(100, 1) * 2

df = pd.DataFrame({"X": X.flatten(), "y": y.flatten()})
print(df)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["X"], y=df["y"], color="blue")
plt.xlabel("X (Independent Variable)")
plt.ylabel("y (Dependent Variable)")
plt.title("Scatter Plot of Dataset Before Model Training")
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Intercept: {model.intercept_[0]:.4f}, Coefficient: {model.coef_[0][0]:.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R-squared: {r2_score(y_test, y_pred):.4f}")

plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# ================================
# PART 2 : MULTIPLE LINEAR REGRESSION
# ================================

X_multi = np.hstack((X, np.random.rand(100, 1) * 5))

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred_multi = model_multi.predict(X_test)

print(f"\nMultiple Regression Coefficients: {model_multi.coef_}")
print(f"Multiple Regression R-squared: {r2_score(y_test, y_pred_multi):.4f}")
