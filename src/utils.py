"""
Utility functions for distributed k-means clustering
"""

import numpy as np
from typing import Tuple, Dict, Any


def calculate_inertia(data: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate inertia (within-cluster sum of squares).
    
    Parameters:
    -----------
    data : np.ndarray
        Data points (N, D)
    centroids : np.ndarray
        Cluster centroids (K, D)
    labels : np.ndarray
        Cluster assignments (N,)
    
    Returns:
    --------
    float
        Sum of squared distances from points to assigned centroids
    """
    inertia = 0.0
    for k in range(len(centroids)):
        mask = labels == k
        if np.sum(mask) > 0:
            distances = np.linalg.norm(data[mask] - centroids[k], axis=1)
            inertia += np.sum(distances ** 2)
    return inertia


def silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate silhouette score for clustering quality.
    Simplified version (computationally intensive).
    
    Parameters:
    -----------
    data : np.ndarray
        Data points (N, D)
    labels : np.ndarray
        Cluster assignments (N,)
    
    Returns:
    --------
    float
        Average silhouette coefficient [-1, 1]
    """
    n_samples = len(data)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    if k < 2:
        return 0.0
    
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Distance to points in same cluster
        same_cluster = labels == labels[i]
        if np.sum(same_cluster) > 1:
            a_i = np.mean(np.linalg.norm(data[same_cluster] - data[i], axis=1))
        else:
            a_i = 0.0
        
        # Minimum distance to other clusters
        b_i = np.inf
        for c in unique_labels:
            if c != labels[i]:
                other_cluster = labels == c
                if np.sum(other_cluster) > 0:
                    min_dist = np.min(np.linalg.norm(data[other_cluster] - data[i], axis=1))
                    b_i = min(b_i, min_dist)
        
        if b_i == np.inf:
            silhouette_vals[i] = 0.0
        else:
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
    
    return np.mean(silhouette_vals)


def davies_bouldin_index(data: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Davies-Bouldin Index for clustering quality.
    Lower values indicate better clustering.
    
    Parameters:
    -----------
    data : np.ndarray
        Data points (N, D)
    centroids : np.ndarray
        Cluster centroids (K, D)
    labels : np.ndarray
        Cluster assignments (N,)
    
    Returns:
    --------
    float
        Davies-Bouldin Index
    """
    k = len(centroids)
    distances = np.zeros(k)
    
    # Calculate average distance within each cluster
    for i in range(k):
        mask = labels == i
        if np.sum(mask) > 0:
            distances[i] = np.mean(np.linalg.norm(data[mask] - centroids[i], axis=1))
        else:
            distances[i] = 0.0
    
    # Calculate pairwise distances between centroids
    centroid_distances = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                centroid_distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    
    # Calculate Davies-Bouldin Index
    db_index = 0.0
    for i in range(k):
        max_ratio = 0.0
        for j in range(k):
            if i != j and centroid_distances[i, j] > 0:
                ratio = (distances[i] + distances[j]) / centroid_distances[i, j]
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    
    return db_index / k if k > 0 else 0.0


def centroid_movement(old_centroids: np.ndarray, new_centroids: np.ndarray) -> float:
    """
    Calculate maximum centroid movement.
    
    Parameters:
    -----------
    old_centroids : np.ndarray
        Previous centroids (K, D)
    new_centroids : np.ndarray
        Updated centroids (K, D)
    
    Returns:
    --------
    float
        Maximum L2 distance between corresponding centroids
    """
    movements = np.linalg.norm(new_centroids - old_centroids, axis=1)
    return np.max(movements)


def compute_statistics(data: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> Dict[str, Any]:
    """
    Compute comprehensive clustering statistics.
    
    Parameters:
    -----------
    data : np.ndarray
        Data points (N, D)
    labels : np.ndarray
        Cluster assignments (N,)
    centroids : np.ndarray
        Cluster centroids (K, D)
    
    Returns:
    --------
    dict
        Dictionary containing various clustering metrics
    """
    stats = {
        'inertia': calculate_inertia(data, centroids, labels),
        'n_samples': len(data),
        'n_features': data.shape[1],
        'n_clusters': len(centroids),
        'cluster_sizes': [np.sum(labels == k) for k in range(len(centroids))],
    }
    
    return stats
