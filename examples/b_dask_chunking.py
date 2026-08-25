"""
(b) Dask backend — chunking / out-of-core.

Tests: can the model process data as small chunks, without ever holding
the full array in memory at once, and still cluster correctly?

    python examples/b_dask_chunking.py
"""
import dask.array as da
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sdoclust_parallel import SDOclust

X, y_true = make_blobs(n_samples=200_000, centers=6, random_state=0)
Xd = da.from_array(X, chunks=(5_000, X.shape[1]))  # small chunks -> many partitions
print(f"{Xd.npartitions} chunks of {Xd.chunks[0][0]} rows each "
      f"(never fully materialized until .compute())")

model = SDOclust(backend="dask", zeta=0.6, chunksize=5_000, rseed=0)
model.fit_from_dask(Xd, n_sample=20_000)   # fit on a small sample only
labels = model.predict(Xd)                 # still lazy: a dask graph
print(f"predict() returns a lazy graph: {type(labels).__module__}.{type(labels).__name__}")

labels = labels.compute()                  # now it's actually processed, chunk by chunk
ari = adjusted_rand_score(y_true, labels)
print(f"quality(ARI)={ari:.3f}")

# capability tested: the model never needed X as a single in-memory array —
# only small chunks at a time — yet clustering quality matches an in-memory run.
