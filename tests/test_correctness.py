"""
Correctness tests for distributed k-means clustering
"""

import numpy as np
from mpi4py import MPI
import sys
sys.path.insert(0, '/Users/csathyanarayanan/Documents/personal/mtech/mlops_assignment2')

from src.distributed_kmeans import DistributedKMeans
from src.data_generator import generate_synthetic_data
from src.utils import calculate_inertia
from sklearn.cluster import KMeans


def test_single_process_equivalence():
    """
    Test that single-process MPI kmeans gives same result as sklearn.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("TEST 1: Single Process Equivalence with scikit-learn")
        print("="*60)
        
        # Generate small dataset
        X, true_labels = generate_synthetic_data(
            n_samples=1000,
            n_features=10,
            n_clusters=5,
            random_state=42
        )
        print(f"Generated data: {X.shape}")
    else:
        X = None
    
    # Run distributed k-means
    dkmeans = DistributedKMeans(
        n_clusters=5,
        max_iterations=100,
        tolerance=1e-4,
        random_state=42,
        verbose=False
    )
    dkmeans.fit(X)
    
    if rank == 0:
        # Run sklearn k-means
        sklearn_kmeans = KMeans(
            n_clusters=5,
            max_iter=100,
            tol=1e-4,
            random_state=42,
            n_init=1,
            init='k-means++'
        )
        sklearn_kmeans.fit(X)
        
        results = dkmeans.get_results()
        
        print(f"\nDistributed K-Means:")
        print(f"  Iterations: {results['n_iterations']}")
        print(f"  Inertia: {results['inertia']:.6f}")
        
        print(f"\nscikit-learn K-Means:")
        print(f"  Iterations: {sklearn_kmeans.n_iter_}")
        print(f"  Inertia: {sklearn_kmeans.inertia_:.6f}")
        
        # Check inertia is within reasonable range
        inertia_ratio = results['inertia'] / sklearn_kmeans.inertia_
        print(f"\nInertia Ratio (distributed/sklearn): {inertia_ratio:.4f}")
        
        if 0.95 < inertia_ratio < 1.10:
            print(" TEST PASSED: Inertia values are comparable")
            return True
        else:
            print(" TEST FAILED: Large difference in inertia")
            return False
    
    return None


def test_convergence_behavior():
    """
    Test that clustering converges properly.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("TEST 2: Convergence Behavior")
        print("="*60)
        
        # Generate dataset
        X, true_labels = generate_synthetic_data(
            n_samples=5000,
            n_features=10,
            n_clusters=5,
            random_state=42
        )
        print(f"Generated data: {X.shape}")
    else:
        X = None
    
    # Run with convergence tracking
    dkmeans = DistributedKMeans(
        n_clusters=5,
        max_iterations=50,
        tolerance=1e-4,
        random_state=42,
        verbose=True
    )
    dkmeans.fit(X)
    
    if rank == 0:
        results = dkmeans.get_results()
        print(f"\nConverged in {results['n_iterations']} iterations")
        
        if results['n_iterations'] < 50:
            print(" TEST PASSED: Algorithm converged before max iterations")
            return True
        else:
            print(" TEST WARNING: Did not converge; may need more iterations")
            return False
    
    return None


def test_cluster_assignment():
    """
    Test that cluster assignments are valid.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("TEST 3: Cluster Assignment Validity")
        print("="*60)
        
        # Generate dataset
        X, true_labels = generate_synthetic_data(
            n_samples=2000,
            n_features=10,
            n_clusters=5,
            random_state=42
        )
        print(f"Generated data: {X.shape}")
    else:
        X = None
    
    dkmeans = DistributedKMeans(
        n_clusters=5,
        max_iterations=100,
        tolerance=1e-4,
        random_state=42,
        verbose=False
    )
    dkmeans.fit(X)
    
    if rank == 0:
        results = dkmeans.get_results()
        labels = results['labels']
        
        # Checks
        unique_labels = np.unique(labels)
        n_assigned = len(labels)
        
        print(f"\nCluster assignments:")
        print(f"  Total points: {n_assigned}")
        print(f"  Unique clusters: {len(unique_labels)}")
        print(f"  Cluster distribution: {np.bincount(labels.astype(int))}")
        
        all_valid = True
        if n_assigned != X.shape[0]:
            print(" Not all points assigned")
            all_valid = False
        
        if len(unique_labels) != 5:
            print(" Wrong number of clusters found")
            all_valid = False
        
        if all_valid:
            print(" TEST PASSED: All assignments valid")
            return True
        else:
            print(" TEST FAILED")
            return False
    
    return None


def run_all_tests():
    """
    Run all correctness tests.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("DISTRIBUTED K-MEANS CORRECTNESS TESTS")
        print("="*60)
    
    test_results = []
    
    test_results.append(test_single_process_equivalence())
    comm.Barrier()
    
    test_results.append(test_convergence_behavior())
    comm.Barrier()
    
    test_results.append(test_cluster_assignment())
    comm.Barrier()
    
    if rank == 0:
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        passed = sum(1 for r in test_results if r is True)
        total = len(test_results)
        print(f"Passed: {passed}/{total}")
        
        if passed == total:
            print("\n ALL TESTS PASSED")
        else:
            print(f"\n {total - passed} TESTS FAILED")


if __name__ == "__main__":
    run_all_tests()
