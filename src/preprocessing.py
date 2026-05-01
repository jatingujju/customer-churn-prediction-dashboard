# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

# =========================
# 2. LOAD DATA (FIXED PATH)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "data", "Churn.csv")

df = pd.read_csv(file_path)

# =========================
# 3. BASIC CLEANING
# =========================
# Drop ID
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# =========================
# 4. FEATURE ENGINEERING
# =========================

# Tenure Groups
df['TenureGroup'] = pd.cut(df['tenure'],
                          bins=[0, 12, 24, 48, 72],
                          labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])

# Avg Charges
df['AvgCharges'] = df['TotalCharges'] / (df['tenure'] + 1)

# High Value Customer
df['HighValue'] = (df['MonthlyCharges'] > 70).astype(int)

# Has Support
df['HasSupport'] = ((df['TechSupport'] == 'Yes') &
                    (df['OnlineSecurity'] == 'Yes')).astype(int)

# =========================
# 5. ENCODING
# =========================
categorical_cols = df.select_dtypes(include='object').columns

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# =========================
# 6. SORT (TIME-SAFE SPLIT)
# =========================
df = df.sort_values(by='tenure')

# =========================
# 7. SPLIT DATA
# =========================
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train.drop('Churn', axis=1)
y_train = train['Churn']

X_test = test.drop('Churn', axis=1)
y_test = test['Churn']

# =========================
# 8. SCALING (FIXED - NO LEAKAGE)
# =========================
scaler = StandardScaler()

# Only scale numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Save scaler
joblib.dump(scaler, os.path.join(BASE_DIR, "models", "scaler.pkl"))

# =========================
# 9. SAVE OUTPUT
# =========================
X_train.to_csv(os.path.join(BASE_DIR, "outputs", "X_train.csv"), index=False)
X_test.to_csv(os.path.join(BASE_DIR, "outputs", "X_test.csv"), index=False)

print("✅ Preprocessing & Feature Engineering Done")
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)