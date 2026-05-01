# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "data", "Churn.csv")

df = pd.read_csv(file_path)

# =========================
# CLEANING
# =========================
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# =========================
# FEATURE ENGINEERING
# =========================
df['EngagementScore'] = (
    (df['OnlineSecurity'] == 'Yes').astype(int) +
    (df['OnlineBackup'] == 'Yes').astype(int) +
    (df['DeviceProtection'] == 'Yes').astype(int) +
    (df['TechSupport'] == 'Yes').astype(int) +
    (df['StreamingTV'] == 'Yes').astype(int) +
    (df['StreamingMovies'] == 'Yes').astype(int)
)

df['Recency'] = 1 / (df['tenure'] + 1)
df['UsageIntensity'] = df['MonthlyCharges'] / (df['tenure'] + 1)
df['PriceRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

# =========================
# FEATURES & TARGET
# =========================
X = df.drop('Churn', axis=1)
y = df['Churn']

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# COLUMN TYPES
# =========================
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# =========================
# PREPROCESSOR
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# =========================
# PIPELINE
# =========================
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight='balanced'
    ))
])

# =========================
# TRAIN
# =========================
pipeline.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# =========================
# DEFAULT EVALUATION
# =========================
print("\n📊 Default Threshold (0.5)\n")
print(classification_report(y_test, y_pred))

# =========================
# CUSTOM THRESHOLD (IMPROVE RECALL)
# =========================
threshold = 0.35
y_pred_thresh = (y_proba >= threshold).astype(int)

print(f"\n🔥 Custom Threshold ({threshold})\n")
print(classification_report(y_test, y_pred_thresh))

# =========================
# ROC-AUC
# =========================
auc = roc_auc_score(y_test, y_proba)
print("\n🎯 ROC-AUC Score:", round(auc, 4))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred_thresh)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Custom Threshold)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Save plot
cm_path = os.path.join(BASE_DIR, "images", "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()

# =========================
# SAVE MODEL
# =========================
model_path = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")
joblib.dump(pipeline, model_path)

print("\n✅ Model saved at:", model_path)