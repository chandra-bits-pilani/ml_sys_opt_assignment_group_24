# Distributed K-Means Clustering Implementation

**Group 24 - ML System Optimization Assignment**

Team Members:
- Chandra Sekar S (2024AC05412)
- Karthik Raja S (2024AC05592)
- Prashanth M G (2024AC05669)
- Sumit Yadav (2024AC05691)
- Venkatesan K (2024AC05445)

## Overview

This project implements distributed k-means clustering using MPI4PY, enabling scalable clustering of large datasets across multiple processes. The implementation uses a master-worker architecture with synchronous centroid updates and MPI collective operations for efficient communication.

## Quick Start

### Installation

```bash
# Clone repository (or set up in your workspace)
cd /Users/csathyanarayanan/Documents/personal/mtech/mlops_assignment2

# Create virtual environment
python3 -m venv kmeans_env
source kmeans_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Prerequisites

- **Python**: 3.8+
- **MPI Implementation**: Open MPI or MPICH2
  ```bash
  # macOS
  brew install open-mpi
  ```

### Running the Implementation

**Single run with 4 processes:**
```bash
mpirun -np 4 python scripts/run_single.py --n-samples 100000 --n-features 10 --n-clusters 5 --verbose
```

**Run correctness tests:**
```bash
mpirun -np 4 python tests/test_correctness.py
```

**Run performance benchmarks:**
```bash
mpirun -np 4 python tests/test_scalability.py
```

## Project Structure

```
mlops_assignment2/
 src/
‚    __init__.py                 # Package initialization
‚    distributed_kmeans.py       # Main MPI4PY implementation
‚    data_generator.py           # Dataset generation utilities
‚    utils.py                    # Helper functions
 tests/
‚    __init__.py
‚    test_correctness.py         # Correctness verification
‚    test_scalability.py         # Performance benchmarks
 scripts/
‚    __init__.py
‚    run_single.py               # Main execution script
 group_24.ipynb                  # Jupyter notebook demonstration
 requirements.txt                # Python dependencies
 README.md                       # This file
```

## Implementation Details

### Core Algorithm

The distributed k-means algorithm follows these phases:

1. **Initialization** (Master only):
   - Generate or load dataset
   - Select initial centroids
   - Broadcast centroids to all workers
   - Scatter data partitions

2. **Iterative Clustering**:
   - **Local Assignment**: Each worker computes distances and assignments
   - **Local Aggregation**: Compute partial sums and counts per cluster
   - **Global Aggregation**: MPI_Reduce to collect global statistics
   - **Centroid Update**: Master updates centroids
   - **Broadcast**: Share updated centroids with all workers

3. **Convergence**: Check centroid movement < tolerance

### Key Components

#### `DistributedKMeans` Class
- Main implementation using MPI4PY
- Synchronous execution model
- Supports both fit() and predict()
- Tracks communication and computation time separately

#### Data Generation
- Synthetic datasets: `generate_synthetic_data()`
- Real datasets: Iris, Digits
- Imbalanced clusters option

#### Testing Framework
- **Correctness**: Compare with scikit-learn, verify convergence
- **Scalability**: Strong scaling, weak scaling, sensitivity analysis
- **Performance**: Execution time, communication overhead, speedup

## Performance Characteristics

### Expected Results (on 4-core machine)

| Metric | Value |
|--------|-------|
| **Strong Scaling (N=100K)** | S_4 ˆ 3.5-3.8 |
| **Communication Overhead** | <15% of execution time |
| **Convergence Iterations** | Similar to sequential k-means |
| **Clustering Quality** | Identical to scikit-learn |

### Communication Cost

Per iteration: O(P Ã K Ã D)
- P = number of processes
- K = number of clusters
- D = number of features

## Usage Examples

### Basic Clustering

```python
from mpi4py import MPI
from src.distributed_kmeans import DistributedKMeans
from src.data_generator import generate_synthetic_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    X, labels = generate_synthetic_data(n_samples=100000, n_clusters=5)
else:
    X = None

kmeans = DistributedKMeans(n_clusters=5, max_iterations=100)
kmeans.fit(X)

if rank == 0:
    results = kmeans.get_results()
    print(f"Inertia: {results['inertia']}")
    print(f"Time: {results['execution_time']:.4f}s")
```

### Running Benchmarks

```bash
# Strong scaling test
for np in 1 2 4 8; do
    mpirun -np $np python scripts/run_single.py --n-samples 100000
done

# Weak scaling test
for np in 1 2 4 8; do
    mpirun -np $np python scripts/run_single.py --n-samples $((25000 * np))
done
```

## Implementation Choices

| Aspect | Choice | Justification |
|--------|--------|---------------|
| **Framework** | MPI4PY | Low-level control, realistic communication modeling |
| **Architecture** | Master-worker | Simplicity, consistent centroid updates |
| **Synchronization** | Synchronous | Matches centralized k-means behavior |
| **Collectives** | Reduce + Bcast | Optimal for aggregation pattern |
| **Distance Metric** | Euclidean (L2) | Standard for k-means |
| **Initialization** | Random from data | Reproducible, extensible to k-means++ |

## Testing

### Correctness Tests
- Single process equivalence with scikit-learn
- Convergence behavior verification
- Cluster assignment validity

### Performance Tests
- Strong scaling (fixed N, vary P)
- Weak scaling (vary N and P proportionally)
- Sensitivity to K (number of clusters)

**Run all tests:**
```bash
mpirun -np 4 python tests/test_correctness.py
mpirun -np 4 python tests/test_scalability.py
```

## Scalability Considerations

### Limitations
- Synchronous execution creates barrier overhead
- Communication grows with number of processes
- Load imbalance if cluster assignments are skewed

### Future Improvements
- Asynchronous centroid updates
- Dynamic load balancing
- k-means++ initialization
- GPU acceleration (CUDA)

## References

1. Lloyd, S. (1982). Least Squares Quantization in PCM. IEEE Transactions on Information Theory.
2. Dean, J. & Ghemawat, S. (2004). MapReduce: Simplified Data Processing on Large Clusters. OSDI.
3. Bahmani, B., et al. (2012). Scalable k-Means++. VLDB.
4. MPI4PY Documentation: https://mpi4py.readthedocs.io/

## Troubleshooting

### MPI Not Found
```bash
# Install Open MPI
brew install open-mpi

# Or install MPICH
brew install mpich
```

### Import Errors
```bash
# Ensure proper Python path
export PYTHONPATH=/Users/csathyanarayanan/Documents/personal/mtech/mlops_assignment2:$PYTHONPATH
```

### Performance Issues
- Reduce dataset size for testing
- Check inter-process communication overhead
- Profile with `mpirun -np P python -m cProfile script.py`

## Authors

Group 24 Members - February 2026

## License

For educational purposes (M.Tech MLOps Program)
