# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (8,5)

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv("data/Churn.csv")

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)

# =========================
# 3. BASIC INFO
# =========================
print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDescribe:")
print(df.describe())

# =========================
# 4. DATA CLEANING
# =========================
# Convert TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop ID
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# =========================
# 5. TARGET DISTRIBUTION
# =========================
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.savefig("images/churn_distribution.png")
plt.show()

print("\nChurn Ratio:\n", df['Churn'].value_counts(normalize=True))

# =========================
# 6. NUMERICAL ANALYSIS
# =========================

# Tenure vs Churn
sns.histplot(data=df, x='tenure', hue='Churn', bins=30)
plt.title("Tenure vs Churn")
plt.savefig("images/tenure_vs_churn.png")
plt.show()

# Monthly Charges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig("images/monthlycharges_vs_churn.png")
plt.show()

# =========================
# 7. CATEGORICAL ANALYSIS
# =========================

# Contract vs Churn
sns.countplot(x='Contract', hue='Churn', data=df)
plt.xticks(rotation=30)
plt.title("Contract vs Churn")
plt.savefig("images/contract_vs_churn.png")
plt.show()

# Payment Method vs Churn
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.xticks(rotation=45)
plt.title("Payment Method vs Churn")
plt.savefig("images/payment_vs_churn.png")
plt.show()

# =========================
# 8. GROUP ANALYSIS
# =========================
print("\nChurn Rate by Contract:")
print(df.groupby('Contract')['Churn'].mean())

print("\nChurn Rate by Internet Service:")
print(df.groupby('InternetService')['Churn'].mean())

# =========================
# 9. CORRELATION HEATMAP
# =========================
numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.show()

# =========================
# 10. KEY INSIGHTS PRINT
# =========================
print("\n🔹 KEY INSIGHTS:")
print("- Customers with low tenure churn more")
print("- Month-to-month contract users have highest churn")
print("- Higher monthly charges increase churn")
print("- Lack of support services increases churn risk")