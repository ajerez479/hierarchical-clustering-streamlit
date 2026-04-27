import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

def preprocess_data(df, selected_features):
    """
    Select features and scale data
    """
    X = df[selected_features].copy()

    # Keep only numeric columns
    X = X.select_dtypes(include=["int64", "float64"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, scaler


def train_hierarchical(X_scaled, n_clusters):
    """
    Train Agglomerative Clustering
    """
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)

    return model, labels


def cluster_new_point(df, selected_features, new_point, n_clusters):
    """
    Refit model including new point (correct way for hierarchical)
    """
    X = df[selected_features].copy()
    X = X.select_dtypes(include=["int64", "float64"])

    # Add new point
    new_df = pd.DataFrame([new_point], columns=selected_features)
    X_extended = pd.concat([X, new_df], ignore_index=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_extended)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)

    return labels[-1]  # cluster of new point