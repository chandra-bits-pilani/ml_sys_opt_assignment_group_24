# P2 Implementation - Distributed K-Means Clustering

## Overview

This document describes the complete implementation of distributed k-means clustering using MPI4PY. The implementation translates the theoretical design into a practical, working system that demonstrates distributed machine learning concepts.

---

## 1. Implementation Architecture

### 1.1 System Components

```
åê
Ç          Application Layer (run_single.py)              Ç
Ç         (Command-line interface for users)              Ç
ò
                       Ç
åºê
Ç         DistributedKMeans Class (Main Logic)            Ç
Ç  - fit() : Execute distributed clustering               Ç
Ç  - predict() : Assign new points                        Ç
Ç  - get_results() : Retrieve metrics                     Ç
ò
                       Ç
åºê
Ç      MPI Communication Layer (MPI4PY)                   Ç
Ç  - Bcast (broadcast centroids)                          Ç
Ç  - Reduce (aggregate statistics)                        Ç
Ç  - Barrier (synchronization)                            Ç
Ç  - Scatter (distribute data)                            Ç
ò
                       Ç
åºê
Ç         Utility Functions                               Ç
Ç  - Data generation (data_generator.py)                  Ç
Ç  - Metrics calculation (utils.py)                       Ç
ò
```

### 1.2 Data Flow

**Initialization:**
```
Data Generation (Rank 0)
    Ü
Scatter to all ranks
    Ü
Broadcast initial centroids
    Ü
All ranks ready for iteration
```

**Per Iteration:**
```
Local Distance Computation (All ranks)
    Ü
Local Assignment & Aggregation (All ranks)
    Ü
MPI_Reduce (partial Üí global) (All ranks Üí Rank 0)
    Ü
Centroid Update (Rank 0 only)
    Ü
MPI_Bcast (new centroids) (Rank 0 Üí All ranks)
    Ü
MPI_Barrier (sync) (All ranks)
    Ü
Convergence Check
```

---

## 2. Core Implementation Details

### 2.1 `DistributedKMeans` Class

**Key Methods:**

#### `__init__(...)`
Initializes the clustering model with:
- Number of clusters (K)
- Maximum iterations
- Convergence tolerance
- MPI communicator and rank information

#### `fit(X)`
Main method for distributed clustering:
1. **Data scatter**: Distribute dataset across all ranks
2. **Iteration loop**: Execute clustering algorithm
3. **Convergence check**: Monitor centroid movement
4. **Results computation**: Final metrics and labels

**Pseudocode:**
```python
def fit(self, X):
    # Phase 1: Initialization
    if rank == 0:
        X = load_data(X)
        centroids = select_initial_centroids(X, K)
        data_splits = partition(X, size)
    
    # Broadcast dimensions and scatter data
    local_X = comm.scatter(data_splits)
    comm.Bcast(centroids)
    
    # Phase 2: Iterative clustering
    for iteration in range(MAX_ITERATIONS):
        # Local computation
        distances = compute_distances(local_X, centroids)  # O(N/P √ K √ D)
        labels = argmin(distances)
        local_sums, local_counts = aggregate(local_X, labels)
        
        # Global aggregation
        global_sums = comm.Reduce(local_sums, MPI.SUM)      # O(K √ D)
        global_counts = comm.Reduce(local_counts, MPI.SUM)  # O(K)
        
        # Centroid update (rank 0)
        if rank == 0:
            new_centroids = global_sums / global_counts
            converged = ||new_centroids - centroids|| < TOL
        
        # Synchronize all ranks
        comm.Bcast(centroids)
        converged = comm.bcast(converged)
        comm.Barrier()
        
        if converged:
            break
    
    # Phase 3: Final metrics
    final_labels = comm.gather(labels)
    final_inertia = comm.reduce(compute_inertia())
    
    return self
```

### 2.2 Key Algorithmic Components

#### Distance Computation
```python
distances = np.linalg.norm(local_X[:, np.newaxis] - centroids, axis=2)
# Shape: (N/P, K) - distances from each local point to each centroid
# Time complexity: O(N/P √ K √ D)
```

#### Local Aggregation
```python
local_sums = np.zeros((K, D))
local_counts = np.zeros(K)

for k in range(K):
    mask = labels == k
    if np.sum(mask) > 0:
        local_sums[k] = np.sum(local_X[mask], axis=0)
        local_counts[k] = np.sum(mask)
# Time complexity: O(N/P)
```

#### Global Aggregation
```python
# MPI_Reduce combines partial results
comm.Reduce(local_sums, global_sums, op=MPI.SUM, root=0)
comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)
# Time complexity: O(log P √ K √ D) [tree reduction]
```

#### Centroid Update
```python
new_centroids = np.zeros((K, D))
for k in range(K):
    if global_counts[k] > 0:
        new_centroids[k] = global_sums[k] / global_counts[k]
# Time complexity: O(K √ D) [negligible on single rank]
```

---

## 3. MPI Communication Pattern

### 3.1 Collective Operations Used

| Operation | Purpose | Time |
|-----------|---------|------|
| `Scatter` | Distribute data partitions | O(log P) |
| `Bcast` | Share centroids | O(log P √ K √ D) |
| `Reduce` | Aggregate statistics | O(log P √ K √ D) |
| `Barrier` | Synchronization | O(P) [network dependent] |
| `Gather` | Collect final labels | O(log P √ N) |

### 3.2 Communication Timeline per Iteration

```
Rank 0: [Compute] [Wait] [Reduce] [Update] [Bcast] [Barrier]
Rank 1: [Compute] [Reduce] [Bcast] [Barrier] [Wait]
Rank 2: [Compute] [Reduce] [Bcast] [Barrier] [Wait]

Total iteration time = max(comp_time) + sum(comm_times) + barrier_time
```

---

## 4. Implementation Choices and Justifications

### Choice 1: Synchronous Execution

**Decision**: All ranks synchronize at each iteration via `MPI.Barrier()`

**Justification**:
- Matches behavior of sequential k-means
- Simpler convergence analysis
- Avoids stale centroid issues
- Easier debugging and testing

**Trade-off**:
- Slower process blocks at barriers
- Cannot exploit compute-communication overlap

---

### Choice 2: Centralized Aggregation

**Decision**: Master (rank 0) aggregates all results

**Justification**:
- Single point of centroid truth
- Consistent updates across all workers
- Simplified implementation
- Matches MapReduce paradigm

**Trade-off**:
- Master becomes potential bottleneck for large K
- Communication cost O(P √ K √ D)

---

### Choice 3: Data Partitioning Strategy

**Decision**: Equal-sized partitions using `np.array_split(X, size)`

**Justification**:
- Simple load balancing for balanced datasets
- Efficient NumPy operation
- Predictable communication cost

**Trade-off**:
- May create imbalance if clusters are skewed
- Doesn't adapt to varying computation per rank

---

### Choice 4: Distance Metric

**Decision**: Euclidean (L2) distance: `||x - Œº||ÇÇ`

**Justification**:
- Standard for k-means
- Efficient NumPy vectorization
- Well-understood properties
- Compatible with convergence proofs

---

### Choice 5: Centroid Initialization

**Decision**: Random selection from data points

**Justification**:
- Simple implementation
- Reproducible with random seed
- No preprocessing needed
- Can be extended to k-means++

**Trade-off**:
- May require more iterations than k-means++
- Quality varies with initialization

---

## 5. Code Organization

### 5.1 File Structure

```
src/
 __init__.py              (Package definition)
 distributed_kmeans.py    (Main implementation - ~250 lines)
 data_generator.py        (Data utilities - ~150 lines)
 utils.py                 (Metrics & helpers - ~200 lines)

tests/
 __init__.py
 test_correctness.py      (Verification tests - ~200 lines)
 test_scalability.py      (Performance benchmarks - ~300 lines)

scripts/
 __init__.py
 run_single.py            (Main execution - ~100 lines)
 run_benchmark.py         (Comprehensive benchmarking - ~150 lines)
```

**Total Implementation: ~1500 lines of code**

### 5.2 Dependencies

```
mpi4py          3.0+    MPI Python bindings
numpy           1.19+   Numerical computation
scikit-learn    0.24+   Reference implementation & datasets
matplotlib      3.3+    Visualization (future)
pandas          1.1+    Results analysis
```

---

## 6. Deployment and Execution

### 6.1 Single-Node Execution (Development)

```bash
# 4 processes on single machine
mpirun -np 4 python scripts/run_single.py \
    --n-samples 100000 \
    --n-clusters 5 \
    --n-features 10 \
    --verbose
```

### 6.2 Multi-Node Execution (Future)

```bash
# Requires host file or machine list
mpirun -np 8 -hostfile hosts.txt python scripts/run_single.py
```

### 6.3 Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `--n-samples` | 100,000 | 1K-1M | Total data points |
| `--n-features` | 10 | 2-100 | Feature dimensions |
| `--n-clusters` | 5 | 2-20 | Number of clusters |
| `--max-iter` | 50 | 10-500 | Iteration limit |
| `--tolerance` | 1e-4 | 1e-6 to 1e-2 | Convergence threshold |

---

## 7. Performance Monitoring

### 7.1 Time Breakdown

```python
# Execution time tracked in three categories:
execution_time    = total wall-clock time
computation_time  = distance calc + local aggregation
communication_time = MPI operations (reduce, bcast, barrier)
overhead_pct      = communication_time / execution_time √ 100
```

### 7.2 Metrics Computed

**Clustering Quality**:
- `inertia`: Sum of squared distances to nearest centroid
- `labels`: Cluster assignment for each point
- `centroids`: Final cluster centers

**Performance**:
- `n_iterations`: Actual iterations to convergence
- `speedup`: S_p = T_1 / T_p
- `communication_overhead`: % time in MPI calls

---

## 8. Testing Strategy

### 8.1 Correctness Tests

1. **Single-Process Equivalence**: Compare against scikit-learn KMeans
2. **Convergence Verification**: Check algorithm stops at tolerance
3. **Assignment Validity**: Verify all points assigned to valid clusters

### 8.2 Performance Tests

1. **Strong Scaling**: Fixed N, varying P àà {1, 2, 4, 8}
2. **Weak Scaling**: N àù P, both varying
3. **Sensitivity**: Vary K àà {2, 5, 10, 20}

### 8.3 Test Execution

```bash
# All tests require 4 processes minimum
mpirun -np 4 python tests/test_correctness.py
mpirun -np 4 python tests/test_scalability.py
```

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

1. **Synchronous barriers**: Cannot overlap compute and communication
2. **Master bottleneck**: Aggregation centralized at rank 0
3. **No load balancing**: Assumes balanced cluster sizes
4. **Memory constraints**: Entire dataset must fit in aggregate memory
5. **Limited to one machine**: Tested only on local multi-process setup

### 9.2 Potential Improvements

1. **Asynchronous updates**: Reduce synchronization overhead
2. **Tree-based reduction**: Distribute aggregation
3. **k-means++ initialization**: Better starting centroids
4. **GPU acceleration**: CUDA for distance computation
5. **Multi-node support**: Proper cluster deployment
6. **Streaming**: Mini-batch or online k-means
7. **Skew handling**: Dynamic load balancing

---

## 10. Verification and Validation

### 10.1 How to Verify Correctness

```bash
# Run with small dataset and single process
mpirun -np 1 python scripts/run_single.py --n-samples 1000 --n-clusters 3
# Should match scikit-learn results

# Run with multiple processes
mpirun -np 4 python scripts/run_single.py --n-samples 1000 --n-clusters 3
# Should have similar inertia and iterations
```

### 10.2 How to Verify Performance

```bash
# Script automatically compares times across process counts
bash benchmark_strong_scaling.sh

# Results saved to results/benchmark_results_*.json
```

---

## Conclusion

This implementation provides a complete, working distributed k-means clustering system using MPI4PY. It successfully demonstrates:

 Data parallelism across multiple processes
 Efficient MPI communication patterns
 Convergence to same solution as sequential algorithm
 Measurable performance improvements with multiple processes
 Clear separation of computation and communication overhead

The implementation serves as a foundation for understanding distributed machine learning and can be extended with the improvements listed above.

