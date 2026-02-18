# ==========================================================
# AI-Powered AdTech Marketing Intelligence System
# Segmentation + Churn + Conversion + Budget Optimization
# ==========================================================

# 1️⃣ IMPORT LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import requests
import warnings
warnings.filterwarnings("ignore")

print("Project Running Successfully 🚀")

# ==========================================================
# 2️⃣ LOAD DATA
# ==========================================================

df = pd.read_csv("Mall_Customers.csv")
print(df.head())

# ==========================================================
# 3️⃣ CUSTOMER SEGMENTATION (UNSUPERVISED ML)
# ==========================================================

kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(
    df[["Annual Income (k$)", "Spending Score (1-100)"]]
)

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="tab10"
)
plt.title("Customer Segmentation using KMeans")
plt.show()

# ==========================================================
# 4️⃣ BUSINESS SEGMENT LABELING
# ==========================================================

cluster_mapping = {
    0: "High Value Customers",
    1: "Budget Shoppers",
    2: "Premium Loyal Customers",
    3: "Low Engagement Customers",
    4: "Young High Spenders"
}

df["Customer Segment"] = df["Cluster"].map(cluster_mapping)

print("\nSegment Distribution:")
print(df["Customer Segment"].value_counts())

# ==========================================================
# 5️⃣ CHURN PREDICTION (SUPERVISED ML)
# ==========================================================

df["Churn"] = np.where(df["Spending Score (1-100)"] < 35, 1, 0)

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf_churn = RandomForestClassifier(random_state=42)
rf_churn.fit(X_train, y_train)

churn_pred = rf_churn.predict(X_test)

print("\n=== Churn Model Results ===")
print(classification_report(y_test, churn_pred))
print("Churn ROC AUC:",
      roc_auc_score(y_test, rf_churn.predict_proba(X_test)[:,1]))


plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, churn_pred),
            annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Churn Model")
plt.show()


# ==========================================================
# 6️⃣ CONVERSION PREDICTION (ADTECH FOCUS)
# ==========================================================

df["Conversion"] = np.where(df["Spending Score (1-100)"] > 60, 1, 0)

conv_y = df["Conversion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, conv_y, test_size=0.3, random_state=42
)

rf_conv = RandomForestClassifier(random_state=42)
rf_conv.fit(X_train, y_train)

conv_pred = rf_conv.predict(X_test)

print("\n=== Conversion Model Results ===")
print("Conversion Accuracy:",
      accuracy_score(y_test, conv_pred))

# ==========================================================
# 7️⃣ MULTI-FACTOR BUDGET OPTIMIZATION
# ==========================================================

channels = pd.DataFrame({
    "Channel": ["Google Ads", "Meta Ads", "LinkedIn Ads"],
    "Current Budget": [50000, 40000, 30000],
    "ROI": [3.5, 2.2, 1.8],
    "Conversion Rate": [0.06, 0.04, 0.03]
})

total_budget = 120000

# Score based on ROI × Conversion Rate
channels["Score"] = channels["ROI"] * channels["Conversion Rate"]
total_score = channels["Score"].sum()

channels["Optimized Budget"] = (
    channels["Score"] / total_score
) * total_budget

print("\nBudget Optimization Table:")
print(channels)

plt.figure(figsize=(8,5))
plt.bar(channels["Channel"],
        channels["Current Budget"],
        alpha=0.6,
        label="Current Budget")

plt.bar(channels["Channel"],
        channels["Optimized Budget"],
        alpha=0.6,
        label="Optimized Budget")

plt.legend()
plt.title("Multi-Factor Budget Reallocation (ROI × Conversion)")
plt.show()

# ==========================================================
# 8️⃣ SEGMENT-LEVEL STRATEGY INSIGHTS
# ==========================================================

segment_summary = df.groupby("Customer Segment")[["Churn","Conversion"]].mean()
print("\nSegment Strategy Summary:")
print(segment_summary)

# ==========================================================
# 9️⃣ LLM-STYLE BUSINESS INSIGHTS (SIMULATED)
# ==========================================================

def generate_business_insight():
    print("\n--- AI Strategic Recommendations ---")
    print("• Target High Value Customers with premium campaigns.")
    print("• Use retargeting ads for high churn segments.")
    print("• Increase spend on Google Ads due to higher ROI and conversion impact.")
    print("• Personalize offers for Low Engagement Customers to reduce churn.\n")

generate_business_insight()

# ==========================================================
# 🔟 BASIC API INTEGRATION EXAMPLE
# ==========================================================

try:
    response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
    data = response.json()
    print("Live USD to INR rate:", data["rates"]["INR"])
except:
    print("API call failed (check internet).")

# ==========================================================
# 1️⃣1️⃣ SAVE FINAL OUTPUT
# ==========================================================

df.to_csv("AdTech_Final_Output.csv", index=False)

print("\nFull AdTech Intelligence System Executed Successfully 🚀🔥")
