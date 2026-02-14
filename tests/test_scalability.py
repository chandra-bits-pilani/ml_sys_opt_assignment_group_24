"""
Performance and scalability benchmark tests for distributed k-means
"""

import numpy as np
import time
from mpi4py import MPI
import sys
import json
sys.path.insert(0, '/Users/csathyanarayanan/Documents/personal/mtech/mlops_assignment2')

from src.distributed_kmeans import DistributedKMeans
from src.data_generator import generate_synthetic_data


def benchmark_strong_scaling():
    """
    Strong scaling: Fixed dataset, vary number of processes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parameters
    N = 100000
    D = 10
    K = 5
    MAX_ITER = 20
    
    if rank == 0:
        print("\n" + "="*70)
        print("STRONG SCALING TEST")
        print("="*70)
        print(f"Dataset size: {N} samples, {D} dimensions, {K} clusters")
        print(f"Current run: {size} processes")
        print("="*70)
        
        # Generate data only once (on master)
        X, _ = generate_synthetic_data(
            n_samples=N,
            n_features=D,
            n_clusters=K,
            random_state=42
        )
        print(f"\nGenerated dataset: {X.shape}")
    else:
        X = None
    
    comm.Barrier()
    
    # Run distributed k-means
    dkmeans = DistributedKMeans(
        n_clusters=K,
        max_iterations=MAX_ITER,
        tolerance=1e-4,
        random_state=42,
        verbose=False
    )
    dkmeans.fit(X)
    
    comm.Barrier()
    
    if rank == 0:
        results = dkmeans.get_results()
        
        print(f"\nResults with {size} processes:")
        print(f"  Execution time:    {results['execution_time']:.4f} seconds")
        print(f"  Computation time:  {results['computation_time']:.4f} seconds")
        print(f"  Communication time: {results['communication_time']:.4f} seconds")
        print(f"  Iterations:        {results['n_iterations']}")
        print(f"  Inertia:           {results['inertia']:.6f}")
        
        comm_overhead = (results['communication_time'] / results['execution_time'] * 100)
        print(f"  Communication overhead: {comm_overhead:.2f}%")
        
        return {
            'n_processes': size,
            'n_samples': N,
            'n_features': D,
            'n_clusters': K,
            'execution_time': results['execution_time'],
            'computation_time': results['computation_time'],
            'communication_time': results['communication_time'],
            'iterations': results['n_iterations'],
            'inertia': results['inertia'],
            'comm_overhead_pct': comm_overhead,
        }
    
    return None


def benchmark_weak_scaling():
    """
    Weak scaling: Proportionally increase dataset and processes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parameters: scale data with number of processes
    base_samples = 25000
    N = base_samples * size
    D = 10
    K = 5
    MAX_ITER = 20
    
    if rank == 0:
        print("\n" + "="*70)
        print("WEAK SCALING TEST")
        print("="*70)
        print(f"Base dataset: {base_samples} samples per process")
        print(f"Total samples: {N} (with {size} processes)")
        print(f"Dimensions: {D}, Clusters: {K}")
        print("="*70)
        
        # Generate data
        X, _ = generate_synthetic_data(
            n_samples=N,
            n_features=D,
            n_clusters=K,
            random_state=42
        )
        print(f"\nGenerated dataset: {X.shape}")
    else:
        X = None
    
    comm.Barrier()
    
    # Run distributed k-means
    dkmeans = DistributedKMeans(
        n_clusters=K,
        max_iterations=MAX_ITER,
        tolerance=1e-4,
        random_state=42,
        verbose=False
    )
    dkmeans.fit(X)
    
    comm.Barrier()
    
    if rank == 0:
        results = dkmeans.get_results()
        
        print(f"\nResults with {size} processes and {N} total samples:")
        print(f"  Execution time:    {results['execution_time']:.4f} seconds")
        print(f"  Computation time:  {results['computation_time']:.4f} seconds")
        print(f"  Communication time: {results['communication_time']:.4f} seconds")
        print(f"  Iterations:        {results['n_iterations']}")
        print(f"  Inertia:           {results['inertia']:.6f}")
        
        comm_overhead = (results['communication_time'] / results['execution_time'] * 100)
        print(f"  Communication overhead: {comm_overhead:.2f}%")
        
        return {
            'n_processes': size,
            'samples_per_process': base_samples,
            'total_samples': N,
            'n_features': D,
            'n_clusters': K,
            'execution_time': results['execution_time'],
            'computation_time': results['computation_time'],
            'communication_time': results['communication_time'],
            'iterations': results['n_iterations'],
            'inertia': results['inertia'],
            'comm_overhead_pct': comm_overhead,
        }
    
    return None


def benchmark_sensitivity():
    """
    Test sensitivity to number of clusters.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Fixed parameters
    N = 50000
    D = 10
    MAX_ITER = 20
    
    if rank == 0:
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS: Varying Number of Clusters")
        print("="*70)
        print(f"Dataset size: {N} samples, {D} dimensions")
        print(f"Processes: {size}")
        print("="*70)
        
        # Generate data once
        X, _ = generate_synthetic_data(
            n_samples=N,
            n_features=D,
            n_clusters=5,
            random_state=42
        )
        print(f"\nGenerated dataset: {X.shape}")
    else:
        X = None
    
    results_list = []
    
    for K in [2, 5, 10, 20]:
        if rank == 0:
            print(f"\n--- Testing with K={K} clusters ---")
        
        comm.Barrier()
        
        dkmeans = DistributedKMeans(
            n_clusters=K,
            max_iterations=MAX_ITER,
            tolerance=1e-4,
            random_state=42,
            verbose=False
        )
        dkmeans.fit(X)
        
        comm.Barrier()
        
        if rank == 0:
            results = dkmeans.get_results()
            print(f"  Execution time: {results['execution_time']:.4f}s")
            print(f"  Iterations: {results['n_iterations']}")
            print(f"  Inertia: {results['inertia']:.6f}")
            
            results_list.append({
                'n_clusters': K,
                'execution_time': results['execution_time'],
                'iterations': results['n_iterations'],
                'inertia': results['inertia'],
            })
    
    if rank == 0:
        return results_list
    
    return None


def run_benchmarks():
    """
    Run all benchmark tests.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*70)
        print("DISTRIBUTED K-MEANS PERFORMANCE BENCHMARKS")
        print("="*70)
    
    # Strong scaling
    strong_result = benchmark_strong_scaling()
    comm.Barrier()
    
    # Weak scaling
    weak_result = benchmark_weak_scaling()
    comm.Barrier()
    
    # Sensitivity
    sensitivity_results = benchmark_sensitivity()
    comm.Barrier()
    
    # Summary
    if rank == 0:
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        if strong_result:
            print("\nSTRONG SCALING:")
            for key, val in strong_result.items():
                print(f"  {key}: {val}")
        
        if weak_result:
            print("\nWEAK SCALING:")
            for key, val in weak_result.items():
                print(f"  {key}: {val}")
        
        if sensitivity_results:
            print("\nSENSITIVITY (K variation):")
            for res in sensitivity_results:
                print(f"  K={res['n_clusters']}: time={res['execution_time']:.4f}s, "
                      f"iter={res['iterations']}, inertia={res['inertia']:.6f}")


if __name__ == "__main__":
    run_benchmarks()
