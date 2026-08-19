import pandas as pd
from scipy.stats import chi2_contingency

# --------------------------------
# 1. Load the Titanic dataset
# --------------------------------

titanic = pd.read_csv("train.csv")


# --------------------------------
# 2. Check the selected columns
# --------------------------------

print(titanic[["Sex", "Survived"]].head())

print(titanic[["Sex", "Survived"]].isnull().sum())

print(titanic["Sex"].unique())

print(titanic["Survived"].unique())

print(len(titanic))


# --------------------------------
# 3. Create the contingency table
# --------------------------------

table = pd.crosstab(
    titanic["Sex"],
    titanic["Survived"]
)

print("\nContingency Table:")
print(table)


# --------------------------------
# 4. Perform Chi-Square Test
# --------------------------------

chi2, p, dof, expected = chi2_contingency(table)

print("\nChi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)

print("\nExpected Frequencies:")
print(expected)


# --------------------------------
# 5. Hypothesis Decision
# --------------------------------

alpha = 0.05

if p < alpha:
    print("\nReject H0")
    print("There is a statistically significant association between gender and survival.")
else:
    print("\nFail to reject H0")
    print("There is no statistically significant association between gender and survival.")
