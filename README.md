# SDO/SDOclust-parallel

Parallel (joblib) and distributed (Dask) implementation of SDO / SDOclust,
based on [CN-TU/pysdoclust](https://github.com/CN-TU/pysdoclust).

Two backends, same API:
- `backend="numpy"` — in-memory, block-wise ops parallelized via `joblib` (`n_jobs`).
- `backend="dask"` — chunked/out-of-core, works with `dask.array` for datasets larger than memory.

## Install

```bash
pip install git+https://github.com/CN-TU/parallel-SDO
```

`dask` (`dask[distributed]`, `dask[dataframe]`, `faiss-cpu`, and `pynndescent` are optional -- only needed if you use
`backend="dask"` or `method="faiss"/"pynndescent"`.

Therefore:
```bash
pip install dask
pip install dask[dataframe]
pip install dask[distributed]
...
```


## Usage

```python
from sklearn.datasets import make_blobs
from sdoclust_parallel import SDOclust

X, _ = make_blobs(n_samples=5000, centers=5, cluster_std=0.5)

model = SDOclust(backend="numpy", n_jobs=-1)
labels = model.fit_predict(X)
scores = model.outlierness(X)
```

With Dask, for data too large to fit in memory:

```python
import dask.array as da
Xd = da.from_array(X, chunks=(100000, X.shape[1]))

model = SDOclust(backend="dask")
model.fit_from_dask(Xd, n_sample=50_000)
labels = model.predict(Xd).compute()
```

See `examples/`:
- `joblib_parallel.py` -- vertical parallelization (n_jobs), timing + clustering quality (ARI).
- `dask_chunking.py` -- out-of-core chunking, quality unaffected by never holding the full array in memory.
- `dask_distributed.py` -- distributed emulation via a local multi-process cluster, confirms work is actually split across worker processes.


## Reference


