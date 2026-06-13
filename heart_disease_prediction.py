import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================
# LOAD DATASET
# =====================================

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "heart.csv")
data = pd.read_csv(data_path)

print("\nHeart Disease Prediction System")
print("-" * 50)

# Features and Target
X = data.drop("target", axis=1)
y = data["target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================
# MODEL TRAINING
# =====================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# =====================================
# MODEL EVALUATION
# =====================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 50)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# helper for user input

def safe_int_input(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a whole number.")

# =====================================
# INTERACTIVE PREDICTION
# =====================================

print("\n")
print("=" * 50)
print("HEART DISEASE RISK PREDICTION")
print("=" * 50)

print("\nPlease enter the following details:\n")

age = safe_int_input("Age: ")
sex = safe_int_input("Gender (0 = Female, 1 = Male): ")
cp = safe_int_input("Chest Pain Type (0-3): ")
trestbps = safe_int_input("Blood Pressure (mmHg): ")
chol = safe_int_input("Cholesterol Level: ")
thalach = safe_int_input("Maximum Heart Rate Achieved: ")
exang = safe_int_input("Exercise Induced Angina (0 = No, 1 = Yes): ")

# Automatically fill remaining features

fbs = data["fbs"].mode()[0]
restecg = data["restecg"].mode()[0]
oldpeak = data["oldpeak"].mean()
slope = data["slope"].mode()[0]
ca = data["ca"].mode()[0]
thal = data["thal"].mode()[0]

patient_data = [[
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]]

# Build a DataFrame with the same feature names the scaler was fitted on
patient_dict = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
}

patient_df = pd.DataFrame([patient_dict], columns=X.columns)

# Scale input
patient_scaled = scaler.transform(patient_df)

# Predict
prediction = model.predict(patient_scaled)
probability = model.predict_proba(patient_scaled)

confidence = max(probability[0]) * 100

# =====================================
# DISPLAY RESULT
# =====================================

print("\n")
print("=" * 50)
print("PREDICTION RESULT")
print("=" * 50)

if prediction[0] == 1:
    print("⚠ Heart Disease Risk Detected")
else:
    print("✓ No Significant Heart Disease Risk Detected")

print(f"\nConfidence Score : {confidence:.2f}%")

if confidence >= 85:
    print("Risk Level       : HIGH")
elif confidence >= 70:
    print("Risk Level       : MODERATE")
else:
    print("Risk Level       : LOW")

print("\nDisclaimer:")
print("This prediction is generated using a machine learning model")
print("and is intended for educational purposes only.")
print("Consult a healthcare professional for medical advice.")