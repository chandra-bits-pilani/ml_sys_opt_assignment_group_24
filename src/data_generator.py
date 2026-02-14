"""
Data generation utilities for distributed k-means clustering
"""

import numpy as np
from typing import Tuple
from sklearn.datasets import make_blobs, load_iris, load_digits
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(
    n_samples: int = 100000,
    n_features: int = 10,
    n_clusters: int = 5,
    random_state: int = 42,
    cluster_std: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic clustered data using sklearn's make_blobs.
    
    Parameters:
    -----------
    n_samples : int
        Number of data points
    n_features : int
        Number of features (dimensions)
    n_clusters : int
        Number of ground-truth clusters
    random_state : int
        Random seed for reproducibility
    cluster_std : float
        Standard deviation of clusters
    
    Returns:
    --------
    tuple
        (data, true_labels) where data is (N, D) and true_labels is (N,)
    """
    np.random.seed(random_state)
    data, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )
    
    # Normalize data to [0, 1] range
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    return data.astype(np.float64), true_labels


def generate_iris_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare Iris dataset.
    
    Returns:
    --------
    tuple
        (data, true_labels)
    """
    data, labels = load_iris(return_X_y=True)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data.astype(np.float64), labels


def generate_digits_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare Digits dataset (handwritten digits).
    
    Returns:
    --------
    tuple
        (data, true_labels)
    """
    data, labels = load_digits(return_X_y=True)
    # Normalize to [0, 1]
    data = data / 16.0
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data.astype(np.float64), labels


def create_imbalanced_clusters(
    n_samples: int = 100000,
    n_features: int = 10,
    n_clusters: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate imbalanced clustered data (some clusters larger than others).
    
    Parameters:
    -----------
    n_samples : int
        Total number of data points
    n_features : int
        Number of features
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple
        (data, true_labels)
    """
    np.random.seed(random_state)
    
    # Create imbalanced cluster sizes
    cluster_sizes = np.random.exponential(scale=n_samples/n_clusters, size=n_clusters)
    cluster_sizes = (cluster_sizes / cluster_sizes.sum() * n_samples).astype(int)
    # Adjust last cluster to ensure exact total
    cluster_sizes[-1] = n_samples - cluster_sizes[:-1].sum()
    
    data, labels = make_blobs(
        n_samples=cluster_sizes.tolist(),
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=random_state
    )
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    return data.astype(np.float64), labels
