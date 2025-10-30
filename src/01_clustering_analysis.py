
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
os.makedirs("outputs/figures", exist_ok=True)
data_cols = [
    "ID", "Date_start_contract", "Date_last_renewal", "Date_next_renewal",
    "Date_birth", "Date_driving_licence", "Distribution_channel", "Seniority",
    "Policies_in_force", "Max_policies", "Max_products", "Lapse", "Date_lapse",
    "Payment", "Premium", "Cost_claims_year", "N_claims_year", "N_claims_history",
    "R_Claims_history", "Type_risk", "Area", "Second_driver", "Year_matriculation",
    "Power", "Cylinder_capacity", "Value_vehicle", "N_doors", "Type_fuel",
    "Length", "Weight"
]

path = r"C:\Users\LENOVO\Desktop\AUTO INSURANCE\Dataset of an actual motor vehicle insurance portfolio\Motor vehicle insurance data.csv"

df=pd.read_csv(path,sep=";",header=0)

df.head(10)

# Feature engineering
df["driver_age"] = (
    pd.to_datetime(df["Date_start_contract"]).dt.year
    - pd.to_datetime(df["Date_birth"]).dt.year
)
df["severity"] = df["Cost_claims_year"] / df["N_claims_year"]

# Working subset for clustering

cols_subset = [
    "driver_age",
    "Seniority",
    "Value_vehicle",
    "Power",
    "Cylinder_capacity",
    "Premium",
    "severity",
    "Area",
    "Type_risk",
    "Cost_claims_year",
]

df_cluster = df.loc[:, cols_subset].copy()
df_cluster.head(10)


# filter on claims
df_cluster=df_cluster[df_cluster["Cost_claims_year"]>0].copy()

# EDA
# missing
df_cluster.isna().sum()
print(df_cluster.isna().sum())
print(df_cluster.describe())
# Distributions

cols_to_plot = ["severity", "Value_vehicle", "Premium"]

for col in cols_to_plot:
    plt.figure(figsize=(6, 4))
    plt.hist(df_cluster[col], bins=40, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"outputs/figures/dist_{col}.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


# Log-transform skewed vars
df_cluster[
    [
        "driver_age",
        "Seniority",
        "Value_vehicle",
        "Power",
        "Cylinder_capacity",
        "Premium",
        "severity", 
        "Area",
        "Type_risk",
        "Cost_claims_year"
    ]
].describe()


# Log-transform skewed vars
df_cluster["severity_log"] = np.log1p(df_cluster["severity"])
df_cluster["Premium_log"] = np.log1p(df_cluster["Premium"])

# Drop original skewed variables (keep *_log)
df_cluster = df_cluster.drop(columns=["severity", "Premium"])


# Scale the final set
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)
#convert back to DataFrame for readability
X_scaled = pd.DataFrame(X_scaled, columns=df_cluster.columns)


for col in [
    "driver_age",
    "Seniority",
    "Value_vehicle",
    "Power",
    "Cylinder_capacity",
    "Premium",
    "severity",
    "Area",
    "Type_risk",
    
]:
    col_min = df_cluster[col].min()
    col_max = df_cluster[col].max()
    print(f"{col}: min={col_min}, max={col_max}")


#Choose k: Elbow & Silhouette

inertias = []
k_values = range(2, 10) 

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster SSE)")
plt.title("Elbow Method")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
plt.savefig("outputs/figures/elbow_method.png", dpi=300, bbox_inches="tight")



#Interpretation

The line sharply drops from k=2 to k=5, meaning adding clusters here significantly improves model fit.

After k≈5, the curve flattens — meaning adding more clusters doesn’t help much.
 So, the “elbow” point looks around k = 5 (or possibly 6).


from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(6,4))
plt.plot(k_values, silhouette_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Analysis")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

for k, score in zip(k_values, silhouette_scores):
    print(f"k={k}: silhouette={score:.3f}")
plt.savefig("outputs/figures/silhouette_scores.png", dpi=300, bbox_inches="tight")


#Interpretation

The highest silhouette score is around k = 4, 6, or 7 (≈ 0.159).

This means the data forms well-separated clusters when we split it into 4 – 7 groups.

Usually, you choose the smallest k that gives a strong score → so k = 4 or 5 would be a balanced choice.


# Final Decision

Both methods agree roughly around k = 5 or k = 6, but here’s the reasoning:

Elbow plot: shows the biggest gain up to k ≈ 5, after which improvements flatten.

Silhouette score: highest near k = 4–7, with small variation (peak ≈ 0.159).

Best balanced choice:
k = 5
It captures meaningful segmentation without overfitting or making clusters too small.

#Fit the Final K-Means Model with K=5

from sklearn.cluster import KMeans

k_final = 5
kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init="auto")
df_cluster["cluster"] = kmeans_final.fit_predict(X_scaled)


#Summarize Each Cluster

print(df_cluster.columns.tolist())
print(df_cluster.head())
print(df_cluster.get("cluster") is None)  
cluster_summary = df_cluster.groupby("cluster").mean(numeric_only=True)
print(cluster_summary)


# Recover severity for plotting
df_cluster["severity"] = np.expm1(df_cluster["severity_log"])


#Visuals
df_cluster["severity"]=np.expm1(df_cluster["severity_log"])
# Visualize Clusters
import seaborn as sns
sns.scatterplot(
    x="Value_vehicle", 
    y="severity", 
    hue="cluster", 
    data=df_cluster, 
    palette="tab10"
)
# save it 
plt.savefig("outputs/figures/cluster_value_vs_severity.png", dpi=300, bbox_inches="tight")

df_cluster.groupby("cluster")[["severity", "Value_vehicle", "Premium_log"]].mean()

sns.scatterplot(x="driver_age", y="Premium_log", hue="cluster", data=df_cluster)

plt.savefig("outputs/figures/cluster_age_vs_premiumlog.png", dpi=300, bbox_inches="tight")

sns.scatterplot(x="Power", y="severity_log", hue="cluster", data=df_cluster)
plt.savefig("outputs/figures/cluster_power_vs_severitylog.png", dpi=300, bbox_inches="tight")





