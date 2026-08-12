import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("dataset/insurance_claims_renamed.csv")

# ==========================
# Fill Missing Values
# ==========================
df.fillna("Unknown", inplace=True)


# ==========================
# Encode Categorical Columns
# ==========================

encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ==========================
# Check Fraud Label Count
# ==========================
print("\nFraud Label Count:")
print(df["Fraud_Label"].value_counts())

# ==========================
# Remove Unnecessary Columns
# ==========================
df = df.drop(
    columns=["Claim_ID", "Customer_ID", "Policy_Number"],
    errors="ignore"
)

# ==========================
# Separate Features and Target
# ==========================
X = df[[
    "Customer_Age",
    "Months_As_Customer",
    "Premium_Amount",
    "Claim_Amount",
    "Vehicle_Claim",
    "Property_Claim",
    "Injury_Claim",
    "Incident_Type",
    "Collision_Type",
    "Damage_Severity",
    "Witness_Count",
    "Police_Report",
    "Property_Damage",
    "Incident_Hour_Of_The_Day",
    "Investigation_Required"
]]
y = df["Fraud_Label"]

# ==========================
# Split Dataset
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Random Forest Model
# ==========================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ==========================
# Train Model
# ==========================
model.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================
y_pred = model.predict(X_test)

# ==========================
# Accuracy
# ==========================
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Model Trained Successfully!")
print("==============================")

print(f"\nAccuracy : {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================
# Feature Importance
# ==========================
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# ==========================
# Save Model
# ==========================
joblib.dump(encoders, "saved_model/encoders.pkl")
joblib.dump(model, "saved_model/fraud_model.pkl")

print("\nModel Saved Successfully!")