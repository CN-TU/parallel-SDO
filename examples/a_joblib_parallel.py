"""
(a) joblib backend — vertical parallelization (single machine, many cores).

Tests: does n_jobs actually speed up SDO/SDOclust, and does the result
stay correct (same clustering quality) regardless of n_jobs?

    python examples/a_joblib_parallel.py
"""
import time
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sdoclust_parallel import SDOclust

X, y_true = make_blobs(n_samples=200_000, centers=6, random_state=0)

for n_jobs in (1, -1):
    model = SDOclust(backend="numpy", zeta=0.6, n_jobs=n_jobs, rseed=0)

    t0 = time.perf_counter()
    labels = model.fit_predict(X)
    elapsed = time.perf_counter() - t0

    ari = adjusted_rand_score(y_true, labels)
    print(f"n_jobs={n_jobs:>2}  time={elapsed:5.2f}s  quality(ARI)={ari:.3f}")

# capability tested: wall-clock time should drop with n_jobs=-1 (more cores
# used in parallel), while ARI stays ~constant (parallelism doesn't change
# the result, only how fast it's computed).
