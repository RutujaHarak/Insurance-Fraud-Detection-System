from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = "insurance-fraud-secret-key"
# Load Model and Encoders
model = joblib.load("saved_model/fraud_model.pkl")
encoders = joblib.load("saved_model/encoders.pkl")


# ==========================
# Home Page
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# About Page
# ==========================
@app.route("/about")
def about():
    return render_template("about.html")


# ==========================
# Dashboard Page
# ==========================
@app.route("/dashboard")
def dashboard():

    df = pd.read_csv("dataset/insurance_claims_renamed.csv")

    total_claims = len(df)
    fraud_claims = len(df[df["Fraud_Label"] == "Yes"])
    genuine_claims = len(df[df["Fraud_Label"] == "No"])

    fraud_rate = round((fraud_claims / total_claims) * 100, 2)

    df["Claim_Date"] = pd.to_datetime(df["Claim_Date"])

    monthly_claims = (
        df.groupby(df["Claim_Date"].dt.strftime("%b"))
          .size()
          .reindex(
              ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"],
              fill_value=0
          )
    )

    monthly_labels = monthly_claims.index.tolist()
    monthly_values = monthly_claims.values.tolist()

    incident_data = df["Incident_Type"].value_counts()

    incident_labels = incident_data.index.tolist()
    incident_values = incident_data.values.tolist()

    top_vehicle = df["Auto_Make"].value_counts().head(5)
    vehicle_labels = top_vehicle.index.tolist()
    vehicle_values = top_vehicle.values.tolist()

    print("Total Claims:", total_claims)
    print("Fraud Claims:", fraud_claims)
    print("Genuine Claims:", genuine_claims)
    print("Fraud Rate:", fraud_rate)

    print("Monthly Labels:", monthly_labels)
    print("Monthly Values:", monthly_values)

    print("Incident Labels:", incident_labels)
    print("Incident Values:", incident_values)

    print("Vehicle Labels:", vehicle_labels)
    print("Vehicle Values:", vehicle_values)

    return render_template(
        "dashboard.html",
        total_claims=total_claims,
        fraud_claims=fraud_claims,
        genuine_claims=genuine_claims,
        fraud_rate=fraud_rate,
        monthly_labels=monthly_labels,
        monthly_values=monthly_values,
        incident_labels=incident_labels,
        incident_values=incident_values,
        vehicle_labels=vehicle_labels,
        vehicle_values=vehicle_values
    )
    
# ==========================
# Predict Page
# ==========================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        # Numeric Inputs
        customer_age = float(request.form["Customer_Age"])
        months_as_customer = float(request.form["Months_As_Customer"])
        premium_amount = float(request.form["Premium_Amount"])
        claim_amount = float(request.form["Claim_Amount"])
        vehicle_claim = float(request.form["Vehicle_Claim"])
        property_claim = float(request.form["Property_Claim"])
        injury_claim = float(request.form["Injury_Claim"])

        witness_count = int(request.form["Witness_Count"])
        incident_hour = int(request.form["Incident_Hour_Of_The_Day"])

        # Categorical Inputs
        incident_type = request.form["Incident_Type"]
        collision_type = request.form["Collision_Type"]
        damage_severity = request.form["Damage_Severity"]
        police_report = request.form["Police_Report"]
        property_damage = request.form["Property_Damage"]
        investigation_required = request.form["Investigation_Required"]

        # Encode categorical values
        incident_type = encoders["Incident_Type"].transform([incident_type])[0]
        collision_type = encoders["Collision_Type"].transform([collision_type])[0]
        damage_severity = encoders["Damage_Severity"].transform([damage_severity])[0]
        police_report = encoders["Police_Report"].transform([police_report])[0]
        property_damage = encoders["Property_Damage"].transform([property_damage])[0]
        investigation_required = encoders["Investigation_Required"].transform([investigation_required])[0]

        # Create feature list
        features = [[
            customer_age,
            months_as_customer,
            premium_amount,
            claim_amount,
            vehicle_claim,
            property_claim,
            injury_claim,
            incident_type,
            collision_type,
            damage_severity,
            witness_count,
            police_report,
            property_damage,
            incident_hour,
            investigation_required
        ]]

        # Prediction
        prediction = model.predict(features)

        # Prediction probability
        probability = model.predict_proba(features)

        if prediction[0] == 1:
            result = "🚨 Fraudulent Claim"
            risk = "🔴 HIGH"
            confidence = round(probability[0][1] * 100, 2)
            recommendation = "Verify documents and conduct manual investigation."

        else:
            result = "✅ Genuine Claim"
            risk = "🟢 LOW"
            confidence = round(probability[0][0] * 100, 2)
            recommendation = "Proceed with normal claim verification."

        return render_template(
            "result.html",
            prediction=result,
            risk=risk,
            confidence=confidence,
            recommendation=recommendation
        )

    return render_template("predict.html")


# ==========================
# Run Flask
# ==========================
if __name__ == "__main__":
    app.run(debug=True)