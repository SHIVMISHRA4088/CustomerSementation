# Required Libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Preprocessing
if 'CustomerID' in df.columns:
    df = df.drop(['CustomerID'], axis=1)
if 'Genre' in df.columns:
    df['Genre'] = df['Genre'].map({'Male': 0, 'Female': 1})

# Feature Selection
X = df.select_dtypes(include=['int64', 'float64'])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose number of clusters (e.g., based on Elbow Method)
k = 5  # You can change this as per your result
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Save the trained model and scaler
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ KMeans model and scaler saved successfully.")
