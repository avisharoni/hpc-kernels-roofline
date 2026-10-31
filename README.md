# hpc-kernels-roofline

![tests](https://github.com/avisharoni/hpc-kernels-roofline/actions/workflows/tests.yml/badge.svg)

**Area:** GPU Kernel / ML Systems / Parallel & HPC

Parallel-algorithm patterns and an empirical roofline on CPU with NumPy.

- `scan.py`: Hillis-Steele (O(n log n) work) and Blelloch work-efficient (up-sweep/down-sweep) prefix scans written as data-parallel steps, plus the three-phase chunked scan used on multicore and GPU (local scan, scan of block sums, fix-up)
- `kernels.py`: STREAM-style triad and DGEMM with FLOP/byte accounting, blocked (tiled) matmul, and the roofline bound `min(peak, AI x BW)`

## Run

```bash
pip install -r requirements.txt
python -m pytest -q
PYTHONPATH=. python bench/bench.py
```

## Results (measured on my machine, Oct 2026)

```
machine: x86_64, 8 logical CPUs, numpy BLAS
triad   n=10000000  AI=0.083 flop/B  1.71 GFLOP/s  20.52 GB/s
matmul  n=256       AI=21.333 flop/B  2.79 GFLOP/s  0.13 GB/s
matmul  n=1024      AI=85.333 flop/B  629.34 GFLOP/s  7.38 GB/s
empirical ceilings: ~20.5 GB/s (triad), ~629.3 GFLOP/s (dgemm); ridge AI ~30.7 flop/B
```

## Limitations

Measured once on a shared cloud VM, so your numbers will differ. The n=256 result is an outlier, likely BLAS thread start-up and warm-up, and is reported as measured. The byte counts are idealized (no write-allocate). No GPU code runs here; the scans show the algorithms GPUs use.

MIT licensed.
