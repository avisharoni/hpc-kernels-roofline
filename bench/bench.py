import os, platform
from hpck.kernels import triad, matmul
print(f"machine: {platform.machine()}, {os.cpu_count()} logical CPUs, numpy BLAS")
rows = [triad(10**7), matmul(256), matmul(1024)]
bw = rows[0]["gbs"]; peak = max(r["gflops"] for r in rows)
for r in rows:
    print(f"{r['kernel']:7s} n={r['n']:<9d} AI={r['ai']:.3f} flop/B  {r['gflops']:.2f} GFLOP/s  {r['gbs']:.2f} GB/s")
print(f"empirical ceilings: ~{bw:.1f} GB/s (triad), ~{peak:.1f} GFLOP/s (dgemm); ridge AI ~{peak/bw:.1f} flop/B")
