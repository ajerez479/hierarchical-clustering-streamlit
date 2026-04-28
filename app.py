import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import preprocess_data, hierarchical_model

st.title("Hierarchical Clustering App")

# =========================
# DATA INPUT
# =========================

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# =========================
# FEATURE SELECTION
# =========================
#numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

selected_features = st.multiselect(
    "Select features for clustering",
    df.columns.tolist()
)
# multiple option with unique selection


if len(selected_features) < 2:
    st.warning("Select at least 2 features")
    st.stop()

# =========================
# PARAMETERS
# =========================
n_clusters = st.slider("Number of clusters", 2, 10, 3)
linkage = st.selectbox("Select linkage",['ward','complete','single'])
# =========================
# MODEL TRAINING
# =========================
X = preprocess_data(df, selected_features)
model, labels = hierarchical_model(X, n_clusters, linkage=linkage)

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
