"""Kernels with known FLOP/byte counts for roofline placement."""
import time
import numpy as np

def bench(fn, reps=5):
    fn(); best = float("inf")
    for _ in range(reps):
        t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
    return best

def triad(n):
    a, b, c = np.random.rand(n), np.random.rand(n), np.empty(n)
    def f(): np.multiply(b, 3.0, out=c); np.add(c, a, out=c)
    t = bench(f); flops = 2 * n; bytes_ = 8 * n * 3  # approx: read a,b write c
    return dict(kernel="triad", n=n, seconds=t, gflops=flops / t / 1e9, gbs=bytes_ / t / 1e9, ai=flops / bytes_)

def matmul(n):
    a, b = np.random.rand(n, n), np.random.rand(n, n)
    t = bench(lambda: a @ b); flops = 2 * n ** 3; bytes_ = 8 * 3 * n * n
    return dict(kernel="matmul", n=n, seconds=t, gflops=flops / t / 1e9, gbs=bytes_ / t / 1e9, ai=flops / bytes_)

def blocked_matmul(a, b, bs=64):
    n = a.shape[0]; c = np.zeros((n, b.shape[1]))
    for i in range(0, n, bs):
        for k in range(0, n, bs):
            for j in range(0, b.shape[1], bs):
                c[i:i+bs, j:j+bs] += a[i:i+bs, k:k+bs] @ b[k:k+bs, j:j+bs]
    return c

def roofline(ai, peak_gflops, peak_gbs):
    return min(peak_gflops, ai * peak_gbs)
