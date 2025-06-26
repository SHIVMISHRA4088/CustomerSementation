import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
# App Title
st.title("🛍️ Customer Segmentation Dashboard")

# Sidebar - Upload Dataset
st.sidebar.header("Upload Data or Use Sample")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Load Sample Data
@st.cache_data
def load_sample_data():
    return pd.read_csv("Mall_Customers.csv")

# Load Uploaded or Sample Data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Uploaded file loaded!")
else:
    df = load_sample_data()
    st.info("ℹ️ Using sample dataset: Mall_Customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Preprocessing
if 'CustomerID' in df.columns:
    df = df.drop(['CustomerID'], axis=1)
if 'Genre' in df.columns:
    df['Genre'] = df['Genre'].map({'Male': 0, 'Female': 1})

features = df.select_dtypes(include=[np.number])

# Load trained model
model = joblib.load("kmeans_model.pkl")  # Make sure you saved this in your notebook
scaler = joblib.load("scaler.pkl")       # Optional, if used

# Scale and Predict
scaled_data = scaler.transform(features)
labels = model.predict(scaled_data)
df['Cluster'] = labels

# Evaluation Scores
sil_score = silhouette_score(scaled_data, labels)
db_index = davies_bouldin_score(scaled_data, labels)

st.subheader("Model Evaluation")
st.write(f"**Silhouette Score:** {sil_score:.3f}")
st.write(f"**Davies–Bouldin Index:** {db_index:.3f}")

# Plot Clusters
st.subheader("Customer Segmentation Visualization")

fig, ax = plt.subplots()
sns.scatterplot(x=features.iloc[:, 0], y=features.iloc[:, 1], hue=labels, palette='Set2', s=100)
plt.xlabel(features.columns[0])
plt.ylabel(features.columns[1])
plt.title("Clustered Customers")
st.pyplot(fig)

# Show Clustered Data
st.subheader("Clustered Data")
st.dataframe(df)

