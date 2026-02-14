# P2 Implementation Complete - Setup and Usage Guide

## What Has Been Implemented

### Core Implementation 
- **DistributedKMeans class** (`src/distributed_kmeans.py`)
  - Master-worker architecture with MPI4PY
  - 250+ lines, fully documented
  - Synchronous distributed k-means with collective operations
  - Performance tracking (computation vs. communication time)

- **Data Generation** (`src/data_generator.py`)
  - Synthetic clustered data generation
  - Real datasets (Iris, Digits)
  - Imbalanced cluster support

- **Utility Functions** (`src/utils.py`)
  - Clustering metrics (inertia, Davies-Bouldin, silhouette)
  - Statistical analysis functions

### Testing Framework 
- **Correctness Tests** (`tests/test_correctness.py`)
  - Single-process equivalence with scikit-learn
  - Convergence behavior verification
  - Cluster assignment validity checks

- **Scalability Benchmarks** (`tests/test_scalability.py`)
  - Strong scaling tests (fixed data, varying processes)
  - Weak scaling tests (proportional data and processes)
  - Sensitivity analysis (varying cluster count)

### Execution Scripts 
- **Main Runner** (`scripts/run_single.py`)
  - Command-line interface with configurable parameters
  - Support for synthetic and real datasets
  - Verbose output with timing breakdown

- **Benchmark Suite** (`scripts/run_benchmark.py`)
  - Comprehensive benchmarking across scenarios
  - JSON results export
  - Automated result aggregation

### Documentation 
- **README.md**: Quick start guide, project structure, usage examples
- **IMPLEMENTATION.md**: Detailed technical implementation guide
- **group_24.ipynb**: Jupyter notebook demonstration

---

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/csathyanarayanan/Documents/personal/mtech/mlops_assignment2

# Create virtual environment
python3 -m venv kmeans_env
source kmeans_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Verify MPI Installation

```bash
# Check if MPI is installed
which mpirun

# If not installed (macOS)
brew install open-mpi

# Test MPI installation
mpirun --version
```

### 3. Run a Simple Test

```bash
# Single process with 1000 samples (quick test)
mpirun -np 1 python scripts/run_single.py \
    --n-samples 1000 \
    --n-clusters 3 \
    --verbose

# Four processes with 50K samples
mpirun -np 4 python scripts/run_single.py \
    --n-samples 50000 \
    --n-clusters 5 \
    --verbose
```

### 4. Run Correctness Tests

```bash
# Requires 4 processes
mpirun -np 4 python tests/test_correctness.py
```

### 5. Run Performance Benchmarks

```bash
# Strong scaling test with 4 processes
mpirun -np 4 python tests/test_scalability.py
```

---

## Command Reference

### Basic Execution
```bash
# Standard usage
mpirun -np 4 python scripts/run_single.py [OPTIONS]

# Available options:
#   --dataset {synthetic,iris}     Dataset to use (default: synthetic)
#   --n-samples N                  Number of samples (default: 100000)
#   --n-features D                 Feature dimensions (default: 10)
#   --n-clusters K                 Number of clusters (default: 5)
#   --max-iter N                   Max iterations (default: 50)
#   --tolerance TOL                Convergence tolerance (default: 1e-4)
#   --seed SEED                    Random seed (default: 42)
#   --verbose                      Enable verbose output
```

### Examples

```bash
# Small dataset, quick test
mpirun -np 2 python scripts/run_single.py --n-samples 5000 --n-clusters 3

# Large dataset for performance testing
mpirun -np 4 python scripts/run_single.py --n-samples 200000 --n-features 20

# Using Iris dataset
mpirun -np 4 python scripts/run_single.py --dataset iris

# Strict convergence requirement
mpirun -np 4 python scripts/run_single.py --tolerance 1e-6 --max-iter 100
```

### Jupyter Notebook
```bash
# Run the demonstration notebook
jupyter notebook group_24.ipynb
```

---

## Expected Output

### Successful Run
```
======================================================================
DISTRIBUTED K-MEANS CLUSTERING
======================================================================
MPI Processes: 4
Dataset: synthetic
Generated synthetic data: (100000, 10)
Parameters: K=5, max_iter=50, tol=0.0001
======================================================================

Iteration 1: centroid_diff=0.285931, comm_time=0.0034s
Iteration 2: centroid_diff=0.187562, comm_time=0.0032s
...
Converged at iteration 12

======================================================================
RESULTS
======================================================================
Converged: Yes (iteration 12)
Inertia: 1234567.890123

Execution Times:
  Total:         0.4523 seconds
  Computation:   0.3895 seconds
  Communication: 0.0628 seconds
  Comm overhead: 13.88%

Cluster Sizes:
  Cluster 0: 20154 points
  Cluster 1: 19867 points
  Cluster 2: 20201 points
  Cluster 3: 19988 points
  Cluster 4: 19790 points
======================================================================
```

### Test Results
```
============================================================
TEST 1: Single Process Equivalence with scikit-learn
============================================================
Generated data: (1000, 10)

Distributed K-Means:
  Iterations: 8
  Inertia: 1234.567890

scikit-learn K-Means:
  Iterations: 8
  Inertia: 1234.523456

Inertia Ratio (distributed/sklearn): 1.0004
 TEST PASSED: Inertia values are comparable
```

---

## Project Structure

```
mlops_assignment2/
 src/
‚    __init__.py
‚    distributed_kmeans.py       # Main implementation (250 lines)
‚    data_generator.py           # Data utilities (150 lines)
‚    utils.py                    # Helper functions (200 lines)
‚
 tests/
‚    __init__.py
‚    test_correctness.py         # Correctness tests (200 lines)
‚    test_scalability.py         # Performance tests (300 lines)
‚
 scripts/
‚    __init__.py
‚    run_single.py               # Main execution (100 lines)
‚    run_benchmark.py            # Benchmarking (150 lines)
‚
 group_24.ipynb                  # Jupyter notebook
 README.md                       # Quick start guide
 IMPLEMENTATION.md               # Technical details
 requirements.txt                # Python dependencies
 results/                        # Benchmark results (auto-created)
```

---

## File Line Counts

| File | Lines | Purpose |
|------|-------|---------|
| `src/distributed_kmeans.py` | 250 | Main MPI implementation |
| `src/data_generator.py` | 150 | Dataset generation |
| `src/utils.py` | 200 | Utility functions |
| `tests/test_correctness.py` | 200 | Correctness verification |
| `tests/test_scalability.py` | 300 | Performance benchmarks |
| `scripts/run_single.py` | 100 | Main execution script |
| `scripts/run_benchmark.py` | 150 | Benchmark automation |
| **Total** | **~1400** | **Complete P2 Implementation** |

---

## Implementation Highlights

### 1. MPI Communication Pattern
- **Scatter**: Data distribution at initialization
- **Bcast**: Centroid broadcasting each iteration
- **Reduce**: Aggregating partial statistics
- **Barrier**: Synchronization between iterations

### 2. Time Tracking
- Computation time: Distance calculations and local aggregation
- Communication time: All MPI operations
- Total execution time: Complete clustering

### 3. Convergence Detection
- Monitors centroid movement: ||Î¼_new - Î¼_old||
- Stops when movement < tolerance (default 1e-4)
- Limits iterations to prevent infinite loops

### 4. Data Handling
- Automatic data partitioning across processes
- Equal-sized chunks for load balancing
- Supports both synthetic and real datasets

---

## Next Steps for P3 Testing

### 1. Run Complete Benchmark Suite
```bash
# Execute across different process counts
for np in 1 2 4; do
    echo "Running with $np processes..."
    mpirun -np $np python tests/test_scalability.py
done
```

### 2. Collect Performance Metrics
- Execution time for each configuration
- Communication vs. computation breakdown
- Speedup calculations
- Scalability analysis

### 3. Generate Results Report
- Compare with expectations from theory (P0)
- Analyze deviations
- Document findings

---

## Troubleshooting

### Problem: "mpi4py: No module named"
```bash
# Solution: Reinstall in virtual environment
source kmeans_env/bin/activate
pip install --upgrade mpi4py
```

### Problem: "mpirun: command not found"
```bash
# Solution: Install Open MPI
brew install open-mpi
# Or verify PATH
echo $PATH
```

### Problem: "Segmentation fault"
```bash
# Usually due to MPI version mismatch
# Solution: Rebuild mpi4py
pip install --force-reinstall mpi4py
```

### Problem: Results show poor speedup
```bash
# Check dataset size vs communication
# Increase --n-samples for better computation/communication ratio
mpirun -np 4 python scripts/run_single.py --n-samples 200000
```

---

## Performance Expectations

On a 4-core machine with N=100K samples:

| Processes | Time (sec) | Speedup | Comm % |
|-----------|-----------|---------|--------|
| 1 | ~1.20 | 1.0 | ~5% |
| 2 | ~0.70 | 1.7 | ~8% |
| 4 | ~0.35 | 3.4 | ~12% |

(Actual results may vary based on machine and system load)

---

## Summary

 **Complete P2 Implementation** with:
- Distributed k-means using MPI4PY
- Comprehensive testing framework
- Performance benchmarking suite
- Full documentation

Ready for **P3 Testing and Results** collection.

