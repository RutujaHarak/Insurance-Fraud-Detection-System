import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("dataset/insurance_claims_renamed.csv")

# Count Fraud Labels
fraud_counts = df["Fraud_Label"].value_counts()

print(fraud_counts)

# Labels
labels = ["Genuine", "Fraud"]

# Values
sizes = [
    fraud_counts["No"],
    fraud_counts["Yes"]
]

# Create Pie Chart
plt.figure(figsize=(6,6))

plt.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#4CAF50", "#F44336"]
)

plt.title("Fraud vs Genuine Claims")

# ==========================
# Damage Severity Bar Chart
# ==========================

damage_counts = df["Damage_Severity"].value_counts()

plt.figure(figsize=(8,5))

damage_counts.plot(
    kind="bar",
    color=["green","orange","red"]
)

plt.title("Claims by Damage Severity")
plt.xlabel("Damage Severity")
plt.ylabel("Number of Claims")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("static/graphs/damage_severity.png")

plt.close()

print("Damage Severity Graph Saved!")

# ==========================
# Incident Type Bar Chart
# ==========================

incident_counts = df["Incident_Type"].value_counts()

plt.figure(figsize=(8,5))

incident_counts.plot(
    kind="bar",
    color="steelblue"
)

plt.title("Claims by Incident Type")
plt.xlabel("Incident Type")
plt.ylabel("Number of Claims")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("static/graphs/incident_type.png")

plt.close()

print("Incident Type Graph Saved!")

# Save Graph
plt.savefig("static/graphs/fraud_pie.png")

plt.close()

print("Graph Saved Successfully!")
# ==========================
# Auto Make Chart
# ==========================

auto_counts = df["Auto_Make"].value_counts().head(10)

plt.figure(figsize=(8,5))

auto_counts.plot(
    kind="bar",
    color="purple"
)

plt.title("Top 10 Auto Makes")
plt.xlabel("Auto Make")
plt.ylabel("Number of Claims")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("static/graphs/auto_make.png")

plt.close()

print("Auto Make Graph Saved!")
# ==========================
# Claim Amount Distribution
# ==========================

plt.figure(figsize=(8,5))

plt.hist(
    df["Claim_Amount"],
    bins=20,
    color="skyblue",
    edgecolor="black"
)

plt.title("Claim Amount Distribution")
plt.xlabel("Claim Amount")
plt.ylabel("Number of Claims")

plt.tight_layout()

plt.savefig("static/graphs/claim_amount.png")

plt.close()

print("Claim Amount Graph Saved!")
# ==========================
# Collision Type Analysis
# ==========================

collision_counts = df["Collision_Type"].value_counts()

plt.figure(figsize=(8,5))

collision_counts.plot(
    kind="bar",
    color="tomato"
)

plt.title("Claims by Collision Type")
plt.xlabel("Collision Type")
plt.ylabel("Number of Claims")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("static/graphs/collision_type.png")

plt.close()

print("Collision Type Graph Saved!")