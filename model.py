import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering

def preprocess_data(df, selected_features):
    """
    Select features and scale data
    """
    X = df[selected_features].copy()

    # Keep only numeric columns
    numerical = X.select_dtypes(include=["int64", "float64"]).columns
    categorical = X.select_dtypes(include=["object"]).columns
   

    scaler = StandardScaler()
    encoder = LabelEncoder()
    for col in numerical:
      X[col] = scaler.fit_transform(X[col])
    for col in categorical:
      X[col] = encoder.fit_transform(X[col])
    return X


def hierarchical_model(X, n_clusters,linkage='ward'):
    """
    Train Agglomerative Clustering
    """
    model = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)
    labels = model.fit_predict(X)

    return model, labels


def cluster_new_point(df, selected_features, new_point, n_clusters):
    """
    Refit model including new point (correct way for hierarchical)
    """

    X = df[selected_features].copy()
    # Add new point
    new_df = pd.DataFrame([new_point], columns=selected_features)
    X = pd.concat([X, new_df], ignore_index=True)

    # Keep only numeric columns
    numerical = X.select_dtypes(include=["int64", "float64"]).columns
    categorical = X.select_dtypes(include=["object"]).columns
   

    scaler = StandardScaler()
    encoder = LabelEncoder()
    X[numerical] = scaler.fit_transform(X[numerical])
    X[categorical] = encoder.fit(X[categorical])
    

    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)

    return labels[-1]  # cluster of new point
