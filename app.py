import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import preprocess_data, train_hierarchical, cluster_new_point

st.title("Hierarchical Clustering App")

# =========================
# DATA INPUT
# =========================
option = st.radio("Choose data source:", ["Upload CSV", "Use sample data"])

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

else:
    df = pd.read_csv("data.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# =========================
# FEATURE SELECTION
# =========================
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

selected_features = st.multiselect(
    "Select features for clustering",
    numeric_columns
)

if len(selected_features) < 2:
    st.warning("Select at least 2 features")
    st.stop()

# =========================
# PARAMETERS
# =========================
n_clusters = st.slider("Number of clusters", 2, 10, 3)

# =========================
# MODEL TRAINING
# =========================
X, X_scaled, scaler = preprocess_data(df, selected_features)
model, labels = train_hierarchical(X_scaled, n_clusters)

df["Cluster"] = labels

st.subheader("Clustered Data")
st.dataframe(df)

# =========================
# VISUALIZATION
# =========================
if len(selected_features) == 2:
    fig, ax = plt.subplots()
    ax.scatter(
        df[selected_features[0]],
        df[selected_features[1]],
        c=df["Cluster"]
    )
    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])

    st.pyplot(fig)
else:
    st.info("Select exactly 2 features for visualization")

# =========================
# NEW POINT (ADVANCED)
# =========================
st.subheader("Assign New Data Point")

new_point = []
for feature in selected_features:
    val = st.number_input(f"{feature}", value=0.0)
    new_point.append(val)

if st.button("Assign Cluster"):
    cluster = cluster_new_point(df, selected_features, new_point, n_clusters)
    st.success(f"Assigned to cluster: {cluster}")