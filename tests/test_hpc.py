import numpy as np
from hpck.scan import hillis_steele, blelloch, chunked_parallel_scan
from hpck.kernels import blocked_matmul, roofline, triad

def test_scans_match_cumsum():
    for n in [1, 2, 7, 64, 1000]:
        x = np.random.default_rng(n).random(n)
        assert np.allclose(hillis_steele(x), np.cumsum(x))
        assert np.allclose(blelloch(x), np.concatenate([[0], np.cumsum(x)[:-1]]))
        assert np.allclose(chunked_parallel_scan(x, 3), np.cumsum(x))

def test_blocked_matmul():
    r = np.random.default_rng(0); a, b = r.random((130, 70)), r.random((70, 90))
    assert np.allclose(blocked_matmul(a, b, 32), a @ b)

def test_roofline():
    assert roofline(0.1, 100, 20) == 2.0 and roofline(100, 100, 20) == 100

def test_triad_reports():
    r = triad(10000); assert r["gflops"] > 0 and abs(r["ai"] - 1 / 12) < 1e-12
