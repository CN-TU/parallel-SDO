# sdoclust-parallel

Parallel (joblib) and distributed (Dask) implementation of SDO / SDOclust,
based on [CN-TU/pysdoclust](https://github.com/CN-TU/pysdoclust).

Two backends, same API:
- `backend="numpy"` — in-memory, block-wise ops parallelized via `joblib` (`n_jobs`).
- `backend="dask"` — chunked/out-of-core, works with `dask.array` for datasets larger than memory.

## Install

```bash
pip install -e .
```

`dask`, `faiss-cpu`, and `pynndescent` are optional — only needed if you use
`backend="dask"` or `method="faiss"/"pynndescent"`.

## Usage

```python
from sklearn.datasets import make_blobs
from sdoclust_parallel import SDOclust

X, _ = make_blobs(n_samples=5000, centers=5)

model = SDOclust(backend="numpy", zeta=0.6, n_jobs=-1)
labels = model.fit_predict(X)
scores = model.outlierness(X)
```

With Dask, for data too large to fit in memory:

```python
import dask.array as da
Xd = da.from_array(X, chunks=(100_000, X.shape[1]))

model = SDOclust(backend="dask", zeta=0.6)
model.fit_from_dask(Xd, n_sample=50_000)
labels = model.predict(Xd).compute()
```

See `examples/`:
- `a_joblib_parallel.py` — vertical parallelization (n_jobs), timing + clustering quality (ARI).
- `b_dask_chunking.py` — out-of-core chunking, quality unaffected by never holding the full array in memory.
- `c_dask_distributed.py` — distributed emulation via a local multi-process cluster, confirms work is actually split across worker processes.

## Reference

Iglesias, Zseby, Hartl, Ferreira. *SDOclust: Clustering with Sparse Data
Observers*. SISAP 2023. Original implementation:
[CN-TU/pysdoclust](https://github.com/CN-TU/pysdoclust).

## License

MIT
