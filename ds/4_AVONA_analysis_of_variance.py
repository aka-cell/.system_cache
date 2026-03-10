import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# -----------------------------
# Sample data (three groups)
# -----------------------------
group1 = [23, 20, 25, 22, 21]
group2 = [27, 30, 29, 31, 28]
group3 = [35, 37, 36, 34, 38]

# Combine data into a DataFrame
data = group1 + group2 + group3
groups = ["A"] * len(group1) + ["B"] * len(group2) + ["C"] * len(group3)

df = pd.DataFrame({"score": data, "group": groups})

# -----------------------------
# One-way ANOVA using statsmodels
# -----------------------------
model = ols("score ~ C(group)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\nANOVA Results:")
print(anova_table)

# -----------------------------
# Tukey's HSD Post-hoc Test
# -----------------------------
tukey_result = pairwise_tukeyhsd(endog=df["score"], groups=df["group"], alpha=0.05)

print("\nTukey's HSD Test Results:")
print(tukey_result.summary())
