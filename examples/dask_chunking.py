"""
(b) Dask backend — chunking / out-of-core.
"""

import dask.array as da
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sdoclust_parallel import SDOclust

X, y_true = make_blobs(n_samples=200000, centers=6, cluster_std=0.5, random_state=0)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} dimensions, {len(np.unique(y_true))} clusters")
Xd = da.from_array(X, chunks=(5000, X.shape[1]))  # small chunks -> many partitions
print(f"{Xd.npartitions} chunks of {Xd.chunks[0][0]} rows each never fully materialized until .compute())")

model = SDOclust(backend="dask", zeta=0.6, chunksize=5_000, rseed=0)
model.fit_from_dask(Xd, n_sample=20000)   # fit on a small sample only
labels = model.predict(Xd)                # still lazy: a dask graph
print(f"predict() returns a lazy graph: {type(labels).__module__}.{type(labels).__name__}")

labels = labels.compute()                  # now it's actually processed, chunk by chunk
ari = adjusted_rand_score(y_true, labels)
print(f"quality(ARI)={ari:.3f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=1, alpha=0.5, cmap='viridis')
plt.colorbar(label='predictions')
plt.xlabel("feat. 1")
plt.ylabel("feat. 2")
plt.grid(True)
plt.show()
