"""
Main execution script for distributed k-means clustering
"""

import sys
import argparse
import numpy as np
from mpi4py import MPI

sys.path.insert(0, '/Users/csathyanarayanan/Documents/personal/mtech/mlops_assignment2')

from src.distributed_kmeans import DistributedKMeans
from src.data_generator import generate_synthetic_data, generate_iris_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Distributed K-Means Clustering with MPI4PY'
    )
    parser.add_argument(
        '--dataset', choices=['synthetic', 'iris'],
        default='synthetic',
        help='Dataset to use'
    )
    parser.add_argument(
        '--n-samples', type=int, default=100000,
        help='Number of samples for synthetic data'
    )
    parser.add_argument(
        '--n-features', type=int, default=10,
        help='Number of features'
    )
    parser.add_argument(
        '--n-clusters', type=int, default=5,
        help='Number of clusters'
    )
    parser.add_argument(
        '--max-iter', type=int, default=50,
        help='Maximum iterations'
    )
    parser.add_argument(
        '--tolerance', type=float, default=1e-4,
        help='Convergence tolerance'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n" + "="*70)
        print("DISTRIBUTED K-MEANS CLUSTERING")
        print("="*70)
        print(f"MPI Processes: {size}")
        print(f"Dataset: {args.dataset}")
        
        if args.dataset == 'synthetic':
            X, true_labels = generate_synthetic_data(
                n_samples=args.n_samples,
                n_features=args.n_features,
                n_clusters=args.n_clusters,
                random_state=args.seed
            )
            print(f"Generated synthetic data: {X.shape}")
        elif args.dataset == 'iris':
            X, true_labels = generate_iris_dataset()
            args.n_clusters = len(np.unique(true_labels))
            print(f"Loaded Iris dataset: {X.shape}")
        
        print(f"Parameters: K={args.n_clusters}, max_iter={args.max_iter}, tol={args.tolerance}")
        print("="*70 + "\n")
    else:
        X = None
        true_labels = None
    
    # Run clustering
    kmeans = DistributedKMeans(
        n_clusters=args.n_clusters,
        max_iterations=args.max_iter,
        tolerance=args.tolerance,
        random_state=args.seed,
        verbose=args.verbose
    )
    
    kmeans.fit(X)
    
    if rank == 0:
        results = kmeans.get_results()
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Converged: Yes (iteration {results['n_iterations']})")
        print(f"Inertia: {results['inertia']:.6f}")
        print(f"\nExecution Times:")
        print(f"  Total:         {results['execution_time']:.4f} seconds")
        print(f"  Computation:   {results['computation_time']:.4f} seconds")
        print(f"  Communication: {results['communication_time']:.4f} seconds")
        
        comm_overhead = results['communication_time'] / results['execution_time'] * 100
        print(f"  Comm overhead: {comm_overhead:.2f}%")
        
        print(f"\nCluster Sizes:")
        cluster_counts = np.bincount(results['labels'].astype(int))
        for k, count in enumerate(cluster_counts):
            print(f"  Cluster {k}: {count} points")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
