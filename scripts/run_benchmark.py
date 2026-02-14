"""
Results aggregation and analysis script
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from mpi4py import MPI
import sys
sys.path.insert(0, '/Users/csathyanarayanan/Documents/personal/mtech/mlops_assignment2')

from src.distributed_kmeans import DistributedKMeans
from src.data_generator import generate_synthetic_data


def run_comprehensive_benchmark():
    """
    Run comprehensive benchmark and save results.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_processes': size,
        'strong_scaling': [],
        'weak_scaling': [],
        'sensitivity': []
    }
    
    if rank == 0:
        print("\n" + "="*70)
        print("COMPREHENSIVE BENCHMARK - All Scenarios")
        print("="*70)
    
    # ===== STRONG SCALING TEST =====
    if rank == 0:
        print("\n[1/3] Strong Scaling Test (N=100K, varying processes)...")
    
    N = 100000
    D = 10
    K = 5
    
    if rank == 0:
        X, _ = generate_synthetic_data(n_samples=N, n_features=D, n_clusters=K, random_state=42)
    else:
        X = None
    
    kmeans = DistributedKMeans(n_clusters=K, max_iterations=20, tolerance=1e-4, random_state=42)
    kmeans.fit(X)
    
    if rank == 0:
        res = kmeans.get_results()
        results['strong_scaling'].append({
            'n_processes': size,
            'n_samples': N,
            'execution_time': res['execution_time'],
            'computation_time': res['computation_time'],
            'communication_time': res['communication_time'],
            'speedup': res['execution_time'],  # T_1 reference will be calculated
            'iterations': res['n_iterations'],
            'inertia': res['inertia']
        })
        print(f"  Completed with {size} processes: {res['execution_time']:.4f}s")
    
    comm.Barrier()
    
    # ===== WEAK SCALING TEST =====
    if rank == 0:
        print("\n[2/3] Weak Scaling Test (N per process=25K)...")
    
    N_per_proc = 25000
    N = N_per_proc * size
    
    if rank == 0:
        X, _ = generate_synthetic_data(n_samples=N, n_features=D, n_clusters=K, random_state=42)
        print(f"  Generated {N} samples for {size} processes")
    else:
        X = None
    
    kmeans = DistributedKMeans(n_clusters=K, max_iterations=20, tolerance=1e-4, random_state=42)
    kmeans.fit(X)
    
    if rank == 0:
        res = kmeans.get_results()
        results['weak_scaling'].append({
            'n_processes': size,
            'samples_per_process': N_per_proc,
            'total_samples': N,
            'execution_time': res['execution_time'],
            'computation_time': res['computation_time'],
            'communication_time': res['communication_time'],
            'iterations': res['n_iterations'],
            'inertia': res['inertia']
        })
        print(f"  Completed: {res['execution_time']:.4f}s")
    
    comm.Barrier()
    
    # ===== SENSITIVITY TEST =====
    if rank == 0:
        print("\n[3/3] Sensitivity Test (varying K)...")
    
    N = 50000
    X, _ = generate_synthetic_data(n_samples=N, n_features=D, n_clusters=5, random_state=42)
    
    for K_test in [2, 5, 10, 20]:
        kmeans = DistributedKMeans(n_clusters=K_test, max_iterations=20, tolerance=1e-4, random_state=42)
        kmeans.fit(X if rank == 0 else None)
        
        if rank == 0:
            res = kmeans.get_results()
            results['sensitivity'].append({
                'n_clusters': K_test,
                'n_processes': size,
                'execution_time': res['execution_time'],
                'iterations': res['n_iterations'],
                'inertia': res['inertia'],
                'comm_overhead_pct': (res['communication_time'] / res['execution_time'] * 100)
            })
            print(f"  K={K_test}: {res['execution_time']:.4f}s ({res['n_iterations']} iterations)")
        
        comm.Barrier()
    
    if rank == 0:
        # Save results
        results_dir = Path('/Users/csathyanarayanan/Documents/personal/mtech/mlops_assignment2/results')
        results_dir.mkdir(exist_ok=True)
        
        result_file = results_dir / f'benchmark_results_{size}proc.json'
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*70)
        print("RESULTS SAVED")
        print("="*70)
        print(f"Results file: {result_file}")
        print(f"\nSummary:")
        print(f"  Processes: {size}")
        print(f"  Strong scaling runs: {len(results['strong_scaling'])}")
        print(f"  Weak scaling runs: {len(results['weak_scaling'])}")
        print(f"  Sensitivity tests: {len(results['sensitivity'])}")
        
        return results
    
    return None


if __name__ == "__main__":
    run_comprehensive_benchmark()
