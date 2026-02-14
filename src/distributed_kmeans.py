"""
Main distributed K-Means implementation using MPI4PY
"""

import numpy as np
import time
from typing import Tuple, Dict, Any, Optional
from mpi4py import MPI


class DistributedKMeans:
    """
    Distributed K-Means clustering using MPI4PY.
    
    Uses a master-worker architecture where:
    - Rank 0 (coordinator) initializes centroids and aggregates results
    - All ranks (workers) perform local distance computation and assignment
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize distributed k-means clustering.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Tolerance for convergence (centroid movement threshold)
        random_state : int, optional
            Random seed for reproducibility
        verbose : bool
            Print detailed iteration information
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.verbose = verbose
        
        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Results storage
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iterations = 0
        self.execution_time = 0.0
        self.communication_time = 0.0
        self.computation_time = 0.0
    
    def _kmeans_plusplus_init(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Initialize centroids using k-means++ algorithm for better convergence.
        
        Parameters:
        -----------
        X : np.ndarray
            Data array (N, D)
        n_clusters : int
            Number of clusters
        
        Returns:
        --------
        centroids : np.ndarray
            Initial centroids (K, D)
        """
        n_samples = X.shape[0]
        centroids = []
        
        # Pick first centroid randomly
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])
        
        # Pick remaining centroids with probability proportional to distance squared
        for _ in range(1, n_clusters):
            # Calculate minimum distance to existing centroids for each point
            distances = np.array([
                min([np.linalg.norm(x - c) for c in centroids])
                for x in X
            ])
            
            # Select next centroid with probability proportional to distance squared
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            next_idx = np.searchsorted(cumulative_probs, r)
            centroids.append(X[next_idx])
        
        return np.array(centroids)
    
    def fit(self, X: Optional[np.ndarray] = None) -> 'DistributedKMeans':
        """
        Fit the distributed k-means model.
        
        Parameters:
        -----------
        X : np.ndarray or None
            Data array (N, D) on rank 0; None on other ranks for distributed setup
        
        Returns:
        --------
        self : DistributedKMeans
            Fitted model
        """
        # Start timing
        total_start = time.time()
        
        # ==================================================
        # PHASE 1: INITIALIZATION (Master broadcasts setup)
        # ==================================================
        
        if self.rank == 0:
            if X is None:
                raise ValueError("Data X must be provided on rank 0")
            X = np.asarray(X, dtype=np.float64)
            self.n_samples, self.n_features = X.shape
            
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            # Initialize centroids using k-means++ algorithm
            centroids = self._kmeans_plusplus_init(X, self.n_clusters)
            centroids = centroids.copy()
            
            # Split data for distribution
            data_splits = np.array_split(X, self.size)
        else:
            X = None
            data_splits = None
            centroids = None
        
        # Broadcast problem dimensions
        dims = self.comm.bcast(
            (self.n_samples if self.rank == 0 else None, self.n_features if self.rank == 0 else None),
            root=0
        )
        self.n_samples, self.n_features = dims
        
        # Scatter data to all processes
        local_X = self.comm.scatter(data_splits, root=0)
        local_X = np.asarray(local_X, dtype=np.float64)
        local_n_samples = len(local_X)
        
        # Broadcast initial centroids
        if self.rank == 0:
            self.centroids = centroids.copy()
        else:
            self.centroids = np.empty((self.n_clusters, self.n_features), dtype=np.float64)
        
        self.comm.Bcast(self.centroids, root=0)
        
        # Synchronize all processes
        self.comm.Barrier()
        
        # ==================================================
        # PHASE 2: ITERATIVE CLUSTERING
        # ==================================================
        
        convergence_comm_start = time.time()
        
        for iteration in range(self.max_iterations):
            iter_comp_start = time.time()
            
            # *** LOCAL ASSIGNMENT AND AGGREGATION ***
            # Compute distances and assign points
            distances = np.linalg.norm(local_X[:, np.newaxis] - self.centroids, axis=2)
            local_labels = np.argmin(distances, axis=1)
            
            # Compute partial sums and counts for each cluster
            local_sums = np.zeros((self.n_clusters, self.n_features), dtype=np.float64)
            local_counts = np.zeros(self.n_clusters, dtype=np.float64)
            
            for k in range(self.n_clusters):
                mask = local_labels == k
                if np.sum(mask) > 0:
                    local_sums[k] = np.sum(local_X[mask], axis=0)
                    local_counts[k] = np.sum(mask)
            
            self.computation_time += (time.time() - iter_comp_start)
            
            # *** GLOBAL AGGREGATION (MPI COLLECTIVE OPERATIONS) ***
            iter_comm_start = time.time()
            
            # Reduce: collect partial sums and counts to master
            global_sums = np.zeros((self.n_clusters, self.n_features), dtype=np.float64)
            global_counts = np.zeros(self.n_clusters, dtype=np.float64)
            
            self.comm.Reduce(local_sums, global_sums, op=MPI.SUM, root=0)
            self.comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)
            
            # Master updates centroids
            if self.rank == 0:
                new_centroids = np.zeros((self.n_clusters, self.n_features), dtype=np.float64)
                for k in range(self.n_clusters):
                    if global_counts[k] > 0:
                        new_centroids[k] = global_sums[k] / global_counts[k]
                    else:
                        # Keep old centroid if cluster is empty
                        new_centroids[k] = self.centroids[k]
                
                # Calculate centroid movement for convergence check
                centroid_diff = np.linalg.norm(new_centroids - self.centroids)
                self.centroids = new_centroids
                converged = centroid_diff < self.tolerance
            else:
                converged = None
            
            # Broadcast updated centroids and convergence flag
            self.comm.Bcast(self.centroids, root=0)
            converged = self.comm.bcast(converged, root=0)
            
            iter_comm_time = time.time() - iter_comm_start
            self.communication_time += iter_comm_time
            
            # Synchronize all processes
            self.comm.Barrier()
            
            # Print progress
            if self.verbose and self.rank == 0:
                print(f"Iteration {iteration + 1}: centroid_diff={centroid_diff:.6f}, "
                      f"comm_time={iter_comm_time:.4f}s")
            
            self.n_iterations = iteration + 1
            
            # Check for convergence
            if converged:
                if self.rank == 0 and self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
        
        self.communication_time = self.communication_time
        
        # ==================================================
        # PHASE 3: FINAL ASSIGNMENT AND METRICS
        # ==================================================
        
        final_comp_start = time.time()
        
        # Final assignment
        distances = np.linalg.norm(local_X[:, np.newaxis] - self.centroids, axis=2)
        local_labels = np.argmin(distances, axis=1)
        
        # Compute local inertia (sum of squared distances)
        local_inertia = np.sum(np.min(distances ** 2, axis=1))
        
        # Reduce inertia to master
        global_inertia = None
        if self.rank == 0:
            global_inertia = 0.0
        global_inertia = self.comm.reduce(local_inertia, op=MPI.SUM, root=0)
        
        if self.rank == 0:
            self.inertia = global_inertia
        
        # Gather labels for final results (on rank 0)
        gathered_labels = self.comm.gather(local_labels, root=0)
        if self.rank == 0:
            self.labels = np.concatenate(gathered_labels)
        
        self.computation_time += (time.time() - final_comp_start)
        
        # Total execution time
        total_end = time.time()
        self.execution_time = total_end - total_start
        
        self.comm.Barrier()
        
        return self
    
    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict cluster labels for new data (works on rank 0).
        
        Parameters:
        -----------
        X : np.ndarray
            Data points to predict (N, D)
        
        Returns:
        --------
        np.ndarray
            Cluster labels
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet")
        
        if self.rank != 0:
            return None
        
        X = np.asarray(X, dtype=np.float64)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get clustering results and performance metrics.
        
        Returns:
        --------
        dict
            Dictionary containing results (on rank 0 only)
        """
        if self.rank != 0:
            return None
        
        return {
            'centroids': self.centroids,
            'labels': self.labels,
            'inertia': self.inertia,
            'n_iterations': self.n_iterations,
            'execution_time': self.execution_time,
            'communication_time': self.communication_time,
            'computation_time': self.computation_time,
            'n_processes': self.size,
        }
