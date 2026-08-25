"""
(a) joblib backend — vertical parallelization (single machine, many cores).

"""

import time
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sdoclust_parallel import SDOclust
import numpy as np

X, y_true = make_blobs(n_samples=200000, centers=6, cluster_std=0.5, random_state=0)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} dimensions, {len(np.unique(y_true))} clusters")

for n_jobs in (1, -1):
    model = SDOclust(backend="numpy", n_jobs=n_jobs, rseed=0)

    t0 = time.perf_counter()
    labels = model.fit_predict(X)
    elapsed = time.perf_counter() - t0

    ari = adjusted_rand_score(y_true, labels)
    print(f"n_jobs={n_jobs:>2}  time={elapsed:5.2f}s  quality(ARI)={ari:.3f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_true, s=1, alpha=0.5, cmap='viridis')
plt.colorbar(label='y_true')
plt.xlabel("feat. 1")
plt.ylabel("feat. 2")
plt.grid(True)
plt.show()
