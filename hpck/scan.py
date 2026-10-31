"""Parallel prefix-scan algorithms expressed as data-parallel steps (NumPy)."""
import numpy as np

def hillis_steele(x):
    """Inclusive scan, O(n log n) work, log n steps."""
    y = np.array(x, dtype=np.float64); d = 1
    while d < len(y):
        y = y.copy(); y[d:] = y[d:] + y[:-d]; d *= 2
    return y

def blelloch(x):
    """Exclusive scan, O(n) work: up-sweep + down-sweep. Pads to power of two."""
    x = np.asarray(x, dtype=np.float64); n = len(x)
    m = 1 << max(0, (n - 1).bit_length()); a = np.zeros(m); a[:n] = x
    d = 1
    while d < m:
        a[2*d-1::2*d] += a[d-1::2*d]; d *= 2
    a[-1] = 0; d = m // 2
    while d >= 1:
        t = a[d-1::2*d].copy(); a[d-1::2*d] = a[2*d-1::2*d]; a[2*d-1::2*d] += t; d //= 2
    return a[:n]

def chunked_parallel_scan(x, workers=4):
    """Three-phase scan as used on multicore: local scans, scan of block sums, fix-up."""
    x = np.asarray(x, dtype=np.float64)
    parts = np.array_split(x, workers)
    local = [np.cumsum(p) for p in parts]
    offs = np.concatenate([[0], np.cumsum([l[-1] if len(l) else 0 for l in local])[:-1]])
    return np.concatenate([l + o for l, o in zip(local, offs)])
