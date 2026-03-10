import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# =============================
# PART 1 : TWO SAMPLE T-TEST
# =============================

np.random.seed(42)
sample1 = np.random.normal(10, 2, 30)
sample2 = np.random.normal(12, 2, 30)

t_statistic, p_value = stats.ttest_ind(sample1, sample2)
alpha = 0.05

print("T-Test Results")
print("T-statistic:", round(t_statistic, 2))
print("P-value:", round(p_value, 4))

if p_value < alpha:
    print("Result: Reject Null Hypothesis (Means are different)")
else:
    print("Result: Fail to Reject Null Hypothesis (Means are similar)")

plt.hist(sample1, alpha=0.5, label="Sample 1")
plt.hist(sample2, alpha=0.5, label="Sample 2")
plt.title("Sample Distributions")
plt.legend()
plt.show()

# =============================
# PART 2 : CHI-SQUARE TEST
# =============================

df = sns.load_dataset("mpg")

df["horsepower_cat"] = pd.cut(
    df["horsepower"], bins=[0, 75, 150, 240], labels=["Low", "Medium", "High"]
)
df["year_cat"] = pd.cut(
    df["model_year"], bins=[69, 72, 74, 84], labels=["T1", "T2", "T3"]
)

table = pd.crosstab(df["horsepower_cat"], df["year_cat"])
print("\nContingency Table:\n", table)

chi2, p, dof, expected = stats.chi2_contingency(table)

print("\nChi-Square Results")
print("Chi-square:", round(chi2, 2))
print("P-value:", round(p, 4))

if p < alpha:
    print("Result: Significant relationship exists")
else:
    print("Result: No significant relationship")
