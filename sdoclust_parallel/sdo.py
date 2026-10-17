import numpy as np
from scipy.spatial import distance
from sklearn.utils.validation import check_array
from .clust import ConnectedComponentsClustering

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment

def materialize(X):
    """Convert dask arrays to numpy arrays"""
    import dask.array as da
    if isinstance(X, da.Array):
        return X.compute()
    return X

def sample_size(N, sigma, error, z=1.96):
    """
    Compute sample size for finite populations (mean estimation).

    Parameters
    ----------
    N : int             - Population size.
    sigma : float       - Standard deviation estimate.
    error : float       - Desired margin of error.
    z : float, optional - Z-score for confidence level (default 1.96).

    Returns
    -------
    int     - Sample size.
    """
    n0 = (z * sigma / error) ** 2
    n = n0 / (1 + (n0 - 1) / N)
    return int(np.ceil(n))

def estimate_sigma_pca(X, max_samples=200000, random_state=0):
    """
    Estimate variance of data via PCA.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)   - Input data.
    max_samples : int, optional                  - Max number of samples to use.
    random_state : int, optional                 - Random seed.

    Returns
    -------
    float   - sigma = sqrt(var(PC1) / d) on standardized data (sigma**2 in (0, 1]).
    """
    m, d = X.shape
    n_sample = min(m, max_samples)

    rng = np.random.default_rng(random_state)
    idx = rng.choice(m, size=n_sample, replace=False)
    Xs = X[idx]

    Xs = StandardScaler().fit_transform(Xs)
    Xp = PCA(n_components=1, random_state=random_state).fit_transform(Xs)

    return np.sqrt(np.var(Xp) / d)
    
def model_error(m):
    """Admissible error of the sample-size formula (your formula, unchanged)."""
    return 0.1 / (1 + np.sqrt(np.log10(m)))

def estimate_k(X, max_samples=100000, random_state=0):
    """Returns (k, sigma2). k is Cochran's sample size with sigma^2 = var(PC1)/d."""
    m = X.shape[0]
    sigma = estimate_sigma_pca(X, max_samples=max_samples, random_state=random_state)
    k = sample_size(m, sigma, model_error(m))
    return max(1, min(k, m)), float(sigma ** 2)

def random_sample(X, n, random_state=0):
    """
    Sample n rows from X, supporting numpy or dask arrays/dataframes.

    Parameters
    ----------
    X : ndarray, dask.array, or dask.DataFrame      - Input data.
    n : int                                         - Number of samples to draw.
    random_state : int, optional                    - Random seed.

    Returns
    -------
    ndarray     - Randomly sampled rows of X.
    """
    if isinstance(X, np.ndarray):
        m = X.shape[0]
        rng = np.random.default_rng(random_state)
        idx = rng.choice(m, size=min(n, m), replace=False)
        return X[idx]
    import dask.array as da
    import dask.dataframe as dd
    if isinstance(X, da.Array):
        m = X.shape[0]
        if np.isnan(m):
            X = X.compute_chunk_sizes()
            m = X.shape[0]
        idx = np.sort(np.random.default_rng(random_state).choice(m, size=min(n, m), replace=False))
        return np.asarray(X[idx].compute())
    if isinstance(X, dd.DataFrame):
        m = len(X)
        frac = min(n, m) / m
        return X.sample(frac=frac, random_state=random_state).compute().to_numpy()
    raise TypeError("X must be NumPy array, Dask array, or Dask DataFrame")


def default_chunksize(m, k, max_pairs=20000000):
    return max(1, min(m, max_pairs // max(k, 1)))

def _knn_batches(X, O, knn, max_pairs=20000000):
    X = np.atleast_2d(X)
    knn = min(knn, O.shape[0])
    bs = default_chunksize(X.shape[0], O.shape[0], max_pairs)
    for i in range(0, X.shape[0], bs):
        D = distance.cdist(X[i:i+bs], O)
        idx = np.argpartition(D, knn - 1, axis=1)[:, :knn]
        yield i, np.take_along_axis(D, idx, axis=1), idx

def _neighbor_batches(X, O, knn, method="brute", index=None):
    X = np.atleast_2d(X)
    if method == "brute":
        for i, _, idx in _knn_batches(X, O, knn):
            yield i, idx
    elif method == "faiss":
        _, idx = index.search(X.astype(np.float32), knn)
        yield 0, idx
    else:
        idx, _ = index.query(X, k=knn)
        yield 0, idx

def harmonize_labels(old_l, new_l):
    mask = old_l >= 0
    K = new_l.max() + 1
    mapping = np.full(K, -1, dtype=int)

    if mask.any():
        old_ids = np.unique(old_l[mask])
        C = np.zeros((K, len(old_ids)), dtype=np.int64)
        np.add.at( C, (new_l[mask], np.searchsorted(old_ids, old_l[mask])), 1 )
        rows, cols = linear_sum_assignment(C, maximize=True)
        valid = C[rows, cols] > 0
        mapping[rows[valid]] = old_ids[cols[valid]]
        available = np.setdiff1d( old_ids, mapping[mapping >= 0] )
    else:
        available = np.array([], dtype=int)

    unmatched = np.flatnonzero(mapping == -1)
    n_reuse = min(len(unmatched), len(available))
    mapping[unmatched[:n_reuse]] = available[:n_reuse]
    remaining = unmatched[n_reuse:]
    next_label = old_l.max() + 1 if old_l.size else 0
    mapping[remaining] = np.arange( next_label, next_label + len(remaining) )

    return mapping[new_l]    
        
def _extend_labels_block(X_block, O, ol, knn, method="brute", index=None, return_membership=False):
    """
    Extend cluster labels for a block of data using the nearest observers.

    Returns
    -------
    C_norm : ndarray (n_samples_block, n_labels) or None   - Normalized cluster counts (only if return_membership).
    labels : ndarray (n_samples_block,)                    - Hard labels.
    """
    X_block = np.atleast_2d(X_block)
    knn_eff = min(knn, len(ol))
    n_labels = int(ol.max()) + 1
    onehot = np.eye(n_labels, dtype=np.float32)[ol]
    n = X_block.shape[0]

    labels = np.empty(n, dtype=np.int64)
    C_norm = np.empty((n, n_labels), dtype=np.float32) if return_membership else None

    for i, closest in _neighbor_batches(X_block, O, knn_eff, method, index):
        counts = np.zeros((closest.shape[0], n_labels), dtype=np.float32)
        for j in range(knn_eff):
            counts += onehot[closest[:, j]]
        labels[i:i + closest.shape[0]] = np.argmax(counts, axis=1)
        if return_membership:
            C_norm[i:i + closest.shape[0]] = counts / knn_eff

    return C_norm, labels

def _score_block(X_block, O, x, method="brute", index=None):
    """
    Compute scores for a block of data based on the closest observers

    Parameters
    ----------
    X_block : ndarray, shape (n_samples_block, n_features)  - Block of data points.
    O : ndarray, shape (n_observers, n_features)            - Observer model.
    x : int                                                 - Number of nearest observers to consider.
    method : str, optional                                  - "brute", "faiss", or "pynndescent".
    index : object, optional                                - Precomputed index if applicable.

    Returns
    -------
    scores : ndarray, shape (n_samples_block,)              - Median observers distance per sample.
    """
    x = min(x, O.shape[0])
    X_block = np.atleast_2d(X_block)
    if method == "brute":
        scores = np.empty(X_block.shape[0])
        for i, dist, _ in _knn_batches(X_block, O, x):
            scores[i:i + dist.shape[0]] = np.median(dist, axis=1)
    elif method == "faiss":
        dist, _ = index.search(X_block.astype(np.float32), x)
        scores = np.median(dist, axis=1)
    else:  # pynndescent
        _, dist = index.query(X_block, k=x)
        scores = np.median(dist, axis=1)
    return scores

def _observer_count_block(X_block, O, x, method="brute", index=None):
    """
    Count how often each observer is a nearest neighbor for a block of data.

    Parameters
    ----------
    X_block : ndarray, shape (n_samples_block, n_features)  - A block of data points.
    O : ndarray, shape (n_observers, n_features)            - Observer model.
    x : int                                                 - Number of neighbors to consider.
    method : str, optional                                  - "brute", "faiss", or "pynndescent".
    index : object, optional                                - Precomputed index if applicable.

    Returns
    -------
    counts : ndarray, shape (n_observers,)                  - Number of times each observer observes.
    """
    x = min(x, O.shape[0])
    X_block = np.atleast_2d(X_block)
    k = O.shape[0]
    if method == "brute":
        P = np.zeros(k, dtype=np.int64)
        for _, _, idx in _knn_batches(X_block, O, x):
            P += np.bincount(idx.ravel(), minlength=k)
        return P
    elif method == "faiss":
        _, closest = index.search(X_block.astype(np.float32), x)
    else:
        closest, _ = index.query(X_block, k=x)
    return np.bincount(closest.ravel(), minlength=k)


def _renew_observers(O, ol, X, kmp, keep_old, x, chunksize, n_jobs, method, build_index, rseed):
    """
    Add new observers sampled from X and drop the least-used OLD observers.
    At most (1 - keep_old) of the old model is replaced; new observers are never dropped in this round.
    Returns (observers, labels) where labels are -1 for the new observers.
    """
    m = X.shape[0]
    k_old = O.shape[0]
    k_new = min(max(1, int(m * kmp)), max(1, int(k_old * (1 - keep_old))))

    O_new = random_sample(X, k_new, rseed)
    k_new = O_new.shape[0]
    O_ext = np.vstack((O, O_new))
    old_l = np.concatenate([ol, -np.ones(k_new, dtype=int)])
    index = build_index(O_ext)

    blocks = [X[i:i+chunksize] for i in range(0, m, chunksize)]
    if n_jobs is None or n_jobs == 1:
        P = np.sum([_observer_count_block(b, O_ext, x, method, index) for b in blocks], axis=0)
    else:
        counts = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_observer_count_block)(b, O_ext, x, method, index) for b in blocks )
        P = np.sum(counts, axis=0)

    P = P.astype(float)
    P[k_old:] = np.inf
    toremove = np.argsort(P, kind="stable")[:k_new]
    keep = np.ones(len(O_ext), dtype=bool)
    keep[toremove] = False
    return O_ext[keep], old_l[keep]


class SDO:
    """
    SDO (Sparse Data Observers) algorithm for anomaly detection.
    Unified API for both NumPy and Dask backends. 

    Parameters
    ----------
    x : int                 - Number of nearest observers used for scoring.
    qv : float              - Quantile value for pruning observers during fitting.
    rseed : int             - Random seed for reproducibility.
    k : int, optional       - Number of observers in the model. If None, automatically estimated.
    backend : str           - Which backend to use: "numpy" or "dask" (deafult: "numpy").
    chunksize : int         - Number of samples per block when processing large datasets.
    method : str            - Neighbor search method: "brute", "faiss", or "pynndescent".
    max_samples_pca : int   - Maximum number of samples for PCA-based sigma estimation.
    n_jobs : int, optional  - Number of parallel jobs for block processing.
    backend_kwargs : dict   - Additional arguments passed to the backend constructor.
    """
    def __init__(self, x=5, qv=0.3, rseed=0, k=None, backend="numpy", chunksize=100000, method="brute", max_samples_pca=100000, n_jobs=None, **backend_kwargs ):
        self.backend = backend

        if backend == "numpy": self._impl = SDO_numpy( x=x, qv=qv, rseed=rseed, k=k, chunksize=chunksize, method=method, max_samples_pca=max_samples_pca, n_jobs=n_jobs)
        elif backend == "dask": self._impl = SDO_dask( x=x, qv=qv, rseed=rseed, k=k, chunksize=chunksize, method=method, 
            max_samples_pca=max_samples_pca, n_jobs=n_jobs, **backend_kwargs )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def fit(self, X):    
        """
        Fit the model to the data.
        Param.:      X : array-like, shape (n_samples, n_features)   - Input data for fitting the model.
        Returns:     self : SDO instance
        """    
        return self._impl.fit(X)

    def fit_from_dask(self, X, n_sample=None):
        """
        Fit the model from a Dask dataset.
        Param.:      X : dask array/dataframe           - Input data for fitting the model.
                     n_sample : int, optional           - Number of samples to use.
        Returns:     self : SDO instance
        """
        return self._impl.fit_from_dask(X, n_sample=n_sample)
            
    def predict(self, X, x=None, O=None):
        """
        Predict scores for new data.
        Param.:      X : array-like, shape (n_samples, n_features)   - Input data for prediction.
                     x : int, optional                               - Number of observers to use for scoring.
                     O : array-like, optional                        - Observers set to use.
        Returns:     scores : ndarray
        """
        return self._impl.predict(X, x=x, O=O)

    def fit_predict(self, X, x=None, O=None):
        """
        Fit the model and predict scores.
        Param.:      X : array-like, shape (n_samples, n_features)   - Input data for model fitting and prediction.
                     x : int, optional                               - Number of observers used for scoring.
                     O : array-like, optional                        - Observer set to use.
        Returns:     scores : ndarray
        """
        return self._impl.fit_predict(X, x=x, O=O)

    def get_observers(self):
        """
        Get the observers set of the model.
        Param.:      None
        Returns:     O : ndarray            - Observers set.
        """
        return self._impl.get_observers()

    def set_observers(self, O):
        """
        Set the observers set of the model.
        Param.:      O : ndarray            - Observer set.
        Returns:     self : SDO instance
        """
        return self._impl.set_observers(O)

class SDO_numpy:
    """
    NumPy-based implementation of the SDO algorithm (see above: SDO).
    It operates fully in-memory and processes data in blocks using NumPy arrays.
    """
    def __init__( self, x=5, qv=0.3, rseed=0, k=None, chunksize=100000, method="brute", max_samples_pca=100000, n_jobs=None ):
        self.x = x
        self.qv = qv
        self.rseed = rseed
        self.k = k
        self.chunksize = chunksize
        self.method = method
        self.max_samples_pca = max_samples_pca
        self.n_jobs = n_jobs

        self.O = None
        self.index = None
        self.k_ = None
        self.sigma2_ = None

        self._faiss = None
        self._pynndescent = None

        if method == "faiss":
            import faiss
            self._faiss = faiss
        elif method == "pynndescent":
            import pynndescent
            self._pynndescent = pynndescent


    def fit(self, X):
        # NOTE: joblib uses threads because FAISS / pynndescent indices are not process-safe

        # model size estimation
        m = X.shape[0]
        k = self.k
        if k is None:
            k, self.sigma2_ = estimate_k(X, self.max_samples_pca, self.rseed)
        self.k_ = k

        # observers
        O0 = random_sample(X, k, self.rseed)

        # index
        self.index = None
        if self.method == "faiss":
            self.index = self._faiss.IndexFlatL2(O0.shape[1])
            self.index.add(O0.astype(np.float32))
        elif self.method == "pynndescent":
            self.index = self._pynndescent.NNDescent( O0.astype(np.float32), n_neighbors=self.x, metric="euclidean")

        # counting observations in blocks
        blocks = [X[i:i+self.chunksize] for i in range(0, m, self.chunksize)]
        if self.n_jobs is None or self.n_jobs == 1:
            P = np.sum([_observer_count_block(b, O0, self.x, self.method, self.index) for b in blocks], axis=0)
        else:
            counts = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_observer_count_block)(b, O0, self.x, self.method, self.index) for b in blocks )
            P = np.sum(counts, axis=0)

        # prune observers
        q = np.quantile(P, self.qv)
        self.O = O0[P >= q]

        # rebuild index
        self.index = None
        if self.method == "faiss":
            self.index = self._faiss.IndexFlatL2(self.O.shape[1])
            self.index.add(self.O.astype(np.float32))
        elif self.method == "pynndescent":
            self.index = self._pynndescent.NNDescent( self.O.astype(np.float32), n_neighbors=self.x, metric="euclidean" )

        return self


    def predict(self, X, x=None, O=None):
        if O is None:
            O = self.O
        if x is None:
            x = self.x

        m = X.shape[0]
        y = np.zeros(m)

        blocks = [X[i:i+self.chunksize] for i in range(0, m, self.chunksize)]
                
        if self.n_jobs is None or self.n_jobs == 1:
            y = np.concatenate([_score_block(b, O, x, self.method, self.index) for b in blocks])
        else:
            scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_score_block)(b, O, x, self.method, self.index) for b in blocks )
            y = np.concatenate(scores)

        return y

    def fit_predict(self, X, x=None, O=None):
        self.fit(X)
        return self.predict(X, x=x, O=O)

    def get_observers(self):
        return self.O, self.index

    def set_observers(self, O):
        self.O = O
        return self

class SDO_dask:
    """
    Dask-based implementation of the SDO algorithm (see above: SDO).
    This backend supports distributed and out-of-core computation.
    """
    def __init__(self, x=5, qv=0.3, rseed=0, k=None, chunksize=100000, method="brute", max_samples_pca=100000, n_jobs=None):
        import dask.array as da
        self._da = da

        self.x = x
        self.qv = qv
        self.rseed = rseed
        self.k = k
        self.chunksize = chunksize
        self.method = "brute"  # for clarity, no faiss/pynndescent
        self.max_samples_pca = max_samples_pca
        self.n_jobs = n_jobs

        self.O = None  # observers
        self.k_ = None
        self.sigma2_ = None
        self._client = None  # removed: no client needed here

    def _as_dask_array(self, X):
        """
        Convert input data to a Dask array if necessary.
        Param.:      X : array-like or dask array, shape (n_samples, n_features)   - Input data.
        Returns:     X : dask array
        """
        if isinstance(X, self._da.Array):
            return X
        return self._da.from_array(X.astype(np.float32), chunks=(self.chunksize, X.shape[1]))

    def fit_from_dask(self, X, n_sample=None):
        if n_sample is None:
            n_sample = self.chunksize
        Xs = random_sample(X, n_sample, self.rseed)
        return self.fit(Xs)

    def fit(self, X):
        if isinstance(X, self._da.Array):
            raise TypeError("fit() expects a NumPy array. Use fit_from_dask() for distributed datasets.")

        X = check_array(X, accept_sparse="csr")
        m = X.shape[0]

        # automatic k if not provided
        k = self.k
        if k is None:
            k, self.sigma2_ = estimate_k(X, self.max_samples_pca, self.rseed)
        self.k_ = k

        O0 = random_sample(X, k, self.rseed)

        # compute observer counts in blocks
        blocks = [X[i:i+self.chunksize] for i in range(0, m, self.chunksize)]
        if self.n_jobs is None or self.n_jobs == 1:
            P = np.sum([_observer_count_block(b, O0, self.x, method="brute") for b in blocks], axis=0)
        else:
            counts = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_observer_count_block)(b, O0, self.x, method="brute") for b in blocks
            )
            P = np.sum(counts, axis=0)

        q = np.quantile(P, self.qv)
        self.O = O0[P >= q]
        return self

    def predict(self, X, x=None, O=None):
        da = self._da

        O = self.O if O is None else O
        x = self.x if x is None else x

        Xd = self._as_dask_array(X)

        # map blocks using pure numpy
        def _block_score(X_block, O, x):
            return _score_block(X_block, O, x, method="brute")

        scores = Xd.map_blocks(_block_score, O=O, x=x, dtype=float, drop_axis=1)
        return scores

    def fit_predict(self, X, x=None, O=None):
        if isinstance(X, self._da.Array):
            self.fit_from_dask(X)
        else:
            self.fit(X)
        return self.predict(X, x=x, O=O)

    def get_observers(self):
        return self.O


class SDOclust:
    """
    SDOclust algorithm for clustering.
    Unified API for both NumPy and Dask backends. 

    Parameters
    ----------
    backend : str           - Backend to use: "numpy" or "dask".
    n_jobs : int, optional  - Number of parallel jobs for block processing.
    **kwargs :              - Parameters forwarded to the backend implementation, including SDO parameters and SDOclust-specific ones.

    Notes
    -----
    SDOclust-specific parameters: zeta, chi, chi_min, chi_prop, e
    """

    def __init__(self, backend="numpy", n_jobs=None, **kwargs):
        self.backend = backend

        if backend == "numpy":
            self._impl = SDOclust_numpy(n_jobs=n_jobs, **kwargs)
        elif backend == "dask":
            self._impl = SDOclust_dask(n_jobs=n_jobs, **kwargs)
        else:
            raise ValueError( f"backend='{backend}' not supported. Use 'numpy' or 'dask'.")

    def fit(self, X):
        """
        Fit the model to the data.
        Param.:      X : array-like, shape (n_samples, n_features)   - Input data for fitting the model.
        Returns:     self : SDOclust instance
        """    
        self._impl.fit(X)
        return self

    def fit_from_dask(self, X, n_sample=None):
        """
        Fit the model from a Dask dataset.
        Param.:      X : dask array/dataframe           - Input data for fitting the model.
                     n_sample : int, optional           - Number of samples to use.
        Returns:     self : SDOclust instance
        """
        return self._impl.fit_from_dask(X, n_sample=n_sample)

    def predict(self, X, return_membership=False, xc=None):
        """
        Predict scores for new data.
        Param.:      X : array-like, shape (n_samples, n_features)   - Input data for prediction.
                     return_membership : bool, optional              - If True, also return membership degrees for each cluster.
                     xc : int, optional                               - Number of observers to use for extending labels.
        Returns:     labels : ndarray, shape (n_samples,)            - Cluster labels for each sample, or
                    (membership, labels) : tuple of ndarrays         - If return_membership=True.
        """
        return self._impl.predict( X, return_membership=return_membership, xc=xc )

    def fit_predict(self, X, return_membership=False, xc=None):
        """
        Fit the model and predict labels.
        Param.:      X : array-like, shape (n_samples, n_features)   - Input data for model fitting and prediction.
                     return_membership : bool, optional              - If True, also return membership degrees for each cluster.
                     xc : int, optional                               - Number of observers to use for extending labels.
        Returns:     labels : ndarray, shape (n_samples,)            - Cluster labels for each sample, or
                    (membership, labels) : tuple of ndarrays         - If return_membership=True.
        """
        return self._impl.fit_predict(X, return_membership=return_membership, xc=xc)

    def update(self, X, keep_old=0.6):
        """
        Update the model with new data.
        Param.:      X : array-like, shape (n_samples, n_features)   - New input data.
                     keep_old : float, optional  - Minimum fraction of previous observers kept (default 0.6).
        Returns:     self : SDOclust instance
        """
        self._impl.update(X, keep_old=keep_old)
        return self

    def update_from_dask(self, X, keep_old=0.6, n_sample=None):
        """
        Update the model from a Dask dataset.
        Param.:      X : dask array/dataframe           - New input data.
                     keep_old : float, optional  - Minimum fraction of previous observers kept (default 0.6).
                     n_sample : int, optional           - Number of samples to use.
        Returns:     self : SDOclust instance
        """
        return self._impl.update_from_dask(X, keep_old=keep_old, n_sample=n_sample)

    def update_predict(self, X, return_membership=False, xc=None, keep_old=0.6):
        """
        Update the model and predict cluster labels.
        Param.:      X : array-like, shape (n_samples, n_features)   - New input data.
                     return_membership : bool, optional              - If True, also return membership degrees.
                     xc : int, optional                              - Number of observers to use for extending labels.
                     keep_old : float, optional  - Minimum fraction of previous observers kept (default 0.6).
        Returns:     labels : ndarray, shape (n_samples,)            - Cluster labels for each sample, or
                    (membership, labels) : tuple of ndarrays         - If return_membership=True.
        """
        return self._impl.update_predict(X, return_membership=return_membership, xc=xc, keep_old=keep_old)

    def outlierness(self, X, x=None):
        """
        Compute outlierness scores for new data.
        Param.:      X : array-like, shape (n_samples, n_features)   - Input data.
                     x : int, optional                               - Number of nearest observers used for scoring.
        Returns:     scores : ndarray, shape (n_samples,)            - Outlierness score for each sample.
        """
        return self._impl.outlierness(X, x)

    def get_observers(self):
        """
        Get the observer set of the model.
        Param.:      None
        Returns:     O : ndarray         - Current observer set.
        """
        return self._impl.get_observers()


class SDOclust_numpy:
    """
    NumPy implementation of SDOclust (see above: SDOclust).

    Parameters
    ----------
    zeta : float          - similarity threshold for connecting observers (local -> 0, global -> 1)
    chi : int, optional   - neighborhood size in the observers graph
    chi_min : int         - minimum value of chi when self-adjusted
    chi_prop : float      - chi proportional factor w.r.t. observers size (if chi is None)
    e : int               - minimum number of observers to form a cluster
    xc : int, optional    - observers used for label propagation (default: x from SDO)

    Notes
    -----
    Other parameters are inherited from SDO (see above: SDO).
    """
    def __init__(self, x=5, qv=0.3, zeta=0.6, chi=None, chi_min=8, chi_prop=0.05, e=3, xc=None, chunksize=100000, method="brute", k=None, rseed=5,        max_samples_pca=100000, n_jobs=None):
        self.x = x
        self.qv = qv
        self.zeta = zeta
        self.chi = chi
        self.chi_min = chi_min
        self.chi_prop = chi_prop
        self.e = e
        self.xc = xc if xc is not None else x
        self.chunksize = chunksize
        self.method = method
        self.rseed = rseed
        self.k = k
        self.max_samples_pca = max_samples_pca
        self.n_jobs = n_jobs

        self.sdo = None
        self.O = None
        self.ol = None
        self.index = None
        self.kmp = None
        self.n_clusters = None
        self.k_ = None
        self.sigma2_ = None

    def fit(self, X):
        X = check_array(X, accept_sparse="csr")

        self.sdo = SDO_numpy( x=self.x, qv=self.qv, chunksize=self.chunksize, method=self.method,  max_samples_pca=self.max_samples_pca, k=self.k, rseed=self.rseed, n_jobs=self.n_jobs).fit(X)

        self.k_, self.sigma2_ = self.sdo.k_, self.sdo.sigma2_
        self.O = self.sdo.O
        self.ol = ConnectedComponentsClustering(zeta=self.zeta, chi=self.chi, chi_min=self.chi_min, chi_prop=self.chi_prop, n_jobs=self.n_jobs).fit(self.O).labels_
        self._remove_small_clusters()
        self._relabel_clusters()
        self._build_index()
        self.n_clusters = len(np.unique(self.ol))
        self.kmp = self.O.shape[0] / X.shape[0]
        return self

    def predict(self, X, return_membership=False, xc=None):
        X = check_array(X, accept_sparse="csr")
        xc = xc if xc is not None else self.xc

        blocks = [X[i:i+self.chunksize] for i in range(0, X.shape[0], self.chunksize)]
        args = (self.O, self.ol, xc, self.method, self.index, return_membership)
        if self.n_jobs is None or self.n_jobs == 1:
            results = [_extend_labels_block(b, *args) for b in blocks]
        else:
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_extend_labels_block)(b, *args) for b in blocks )

        labels = np.concatenate([r[1] for r in results])
        if return_membership:
            return np.vstack([r[0] for r in results]), labels
        return labels

    def fit_predict(self, X, return_membership=False, xc=None):
        self.fit(X)
        return self.predict(X, return_membership=return_membership, xc=xc)

    def update(self, X, keep_old=0.6):
        X = check_array(X, accept_sparse="csr")

        self.O, old_l = _renew_observers(self.O, self.ol, X, self.kmp, keep_old, self.x, self.chunksize,
                                         self.n_jobs, self.method, self._build_index_for, self.rseed)

        new_l = ConnectedComponentsClustering(zeta=self.zeta, chi=self.chi, chi_min=self.chi_min, chi_prop=self.chi_prop, n_jobs=self.n_jobs).fit(self.O).labels_
        self.ol = harmonize_labels(old_l, np.asarray(new_l))
        self.n_clusters = len(np.unique(self.ol))
        self._build_index()
        self.sdo.O, self.sdo.index = self.O, self.index
        return self

    def update_predict(self, X, return_membership=False, xc=None, keep_old=0.6):
        self.update(X, keep_old=keep_old)
        return self.predict(X, return_membership=return_membership, xc=xc)

    def outlierness(self, X, x=None):
        x = x if x is not None else self.x
        return self.sdo.predict(X, x)

    def get_observers(self):
        return self.O, self.index

    def _remove_small_clusters(self):
        """
        Remove clusters smaller than the minimum allowed size.
        """
        labels, counts = np.unique(self.ol, return_counts=True)
        toremove = np.zeros(len(self.ol), dtype=bool)

        for l, c in zip(labels, counts):
            if c <= self.e:
                toremove[self.ol == l] = True

        if not np.all(toremove):
            self.O = self.O[~toremove]
            self.ol = self.ol[~toremove]

    def _relabel_clusters(self):
        """
        Relabel clusters consecutively from 0 to n_clusters-1.
        """
        unique = np.unique(self.ol)
        mapping = {old: new for new, old in enumerate(unique)}
        self.ol = np.array([mapping[l] for l in self.ol])

    def _build_index(self):
        """
        Build a neighbor search index for the current observer set.
        """
        self.index = self._build_index_for(self.O)

    def _build_index_for(self, O):
        """
        Build a neighbor search index for a given observers set.
        Param.:     O : ndarray, shape (n_observers, n_features)    - Observers set to build the index for.
        Returns:    index : object or None                          - Index object compatible with the chosen method.
        """
        if self.method == "faiss":
            index = self.sdo._faiss.IndexFlatL2(O.shape[1])
            index.add(O.astype(np.float32))
            return index
        elif self.method == "pynndescent":
            return self.sdo._pynndescent.NNDescent( O.astype(np.float32), n_neighbors=self.xc, metric="euclidean" )
        else:
            return None


class SDOclust_dask:
    """
    Dask implementation of SDOclust (see above: SDOclust).

    Parameters
    ----------
    zeta : float          - similarity threshold for connecting observers (local -> 0, global -> 1)
    chi : int, optional   - neighborhood size in the observers graph
    chi_min : int         - minimum value of chi when self-adjusted
    chi_prop : float      - chi proportional factor w.r.t. observers size (if chi is None)
    e : int               - minimum number of observers to form a cluster
    xc : int, optional    - observers used for label propagation (default: x from SDO)

    Notes
    -----
    Other parameters are inherited from SDO (see above: SDO).
    """

    def __init__(self, x=5, qv=0.3, zeta=0.6, chi=None, chi_min=8, chi_prop=0.05,
                 e=3, xc=None, chunksize=100000, k=None, rseed=5, max_samples_pca=100000, n_jobs=None):
        import dask.array as da
        self._da = da

        self.x = x
        self.qv = qv
        self.zeta = zeta
        self.chi = chi
        self.chi_min = chi_min
        self.chi_prop = chi_prop
        self.e = e
        self.xc = xc if xc is not None else x
        self.chunksize = chunksize
        self.rseed = rseed
        self.k = k
        self.max_samples_pca = max_samples_pca
        self.n_jobs = n_jobs

        self.sdo = None
        self.O = None
        self.ol = None
        self.n_clusters = None
        self.kmp = None
        self.k_ = None
        self.sigma2_ = None

    def _as_dask_array(self, X):
        """
        Convert input data to a Dask array if necessary.
        Param.:      X : array-like or dask array, shape (n_samples, n_features)   - Input data.
        Returns:     X : dask array
        """
        if isinstance(X, self._da.Array):
            return X
        return self._da.from_array(X.astype(np.float32), chunks=(self.chunksize, X.shape[1]))

    def fit_from_dask(self, X, n_sample=None):
        if n_sample is None:
            n_sample = self.chunksize
        Xs = random_sample(X, n_sample, self.rseed)
        return self.fit(Xs)

    def update_from_dask(self, X, n_sample=None, keep_old=0.6):
        if n_sample is None:
            n_sample = self.chunksize
        Xs = random_sample(X, n_sample, self.rseed)
        return self.update(Xs, keep_old=keep_old)

    def fit(self, X):
        X = check_array(X, accept_sparse="csr")

        self.sdo = SDO_dask(
            x=self.x, qv=self.qv, chunksize=self.chunksize, k=self.k,
            rseed=self.rseed, max_samples_pca=self.max_samples_pca, n_jobs=self.n_jobs).fit(X)

        self.k_, self.sigma2_ = self.sdo.k_, self.sdo.sigma2_
        self.O = self.sdo.O

        self.ol = ConnectedComponentsClustering(
            zeta=self.zeta, chi=self.chi, chi_min=self.chi_min, chi_prop=self.chi_prop,
            n_jobs=self.n_jobs).fit(self.O).labels_

        self._remove_small_clusters()
        self._relabel_clusters()
        self.n_clusters = len(np.unique(self.ol))
        self.kmp = self.O.shape[0] / X.shape[0]

        return self

    def update(self, X, keep_old=0.6):
        X = check_array(X, accept_sparse="csr")

        self.O, old_l = _renew_observers(self.O, self.ol, X, self.kmp, keep_old, self.x, self.chunksize,
                                         self.n_jobs, "brute", lambda O: None, self.rseed)

        new_l = ConnectedComponentsClustering( zeta=self.zeta, chi=self.chi, chi_min=self.chi_min, chi_prop=self.chi_prop,
            n_jobs=self.n_jobs).fit(self.O).labels_
        self.ol = harmonize_labels(old_l, np.asarray(new_l))
        self.n_clusters = len(np.unique(self.ol))
        self.sdo.O = self.O

        return self


    def predict(self, X, return_membership=False, xc=None):
        xc = xc if xc is not None else self.xc
        Xd = self._as_dask_array(X)
        n_labels = int(self.ol.max()) + 1

        # map function for blocks
        def _map_labels(block, O, ol, knn):
            return _extend_labels_block(block, O, ol, knn, "brute", None, False)[1]

        def _map_membership(block, O, ol, knn):
            return _extend_labels_block(block, O, ol, knn, "brute", None, True)[0]

        if return_membership:
            Md = Xd.map_blocks(_map_membership, O=self.O, ol=self.ol, knn=xc,
                               dtype=np.float32, chunks=(Xd.chunks[0], (n_labels,)))
            return Md, self._da.argmax(Md, axis=1)

        return Xd.map_blocks(_map_labels, O=self.O, ol=self.ol, knn=xc, dtype=np.int64, drop_axis=1)

    def fit_predict(self, X, return_membership=False, xc=None):
        if isinstance(X, self._da.Array):
            self.fit_from_dask(X)
        else:
            self.fit(X)
        return self.predict(X, return_membership=return_membership, xc=xc)

    def update_predict(self, X, return_membership=False, xc=None, keep_old=0.6):
        self.update_from_dask(X, keep_old=keep_old)
        return self.predict(X, return_membership=return_membership, xc=xc)

    def outlierness(self, X, x=None):
        return self.sdo.predict(X, x)

    def get_observers(self):
        return self.O

    def _remove_small_clusters(self):
        """
        Remove clusters smaller than the minimum allowed size.
        """
        labels, counts = np.unique(self.ol, return_counts=True)
        mask = np.ones(len(self.ol), dtype=bool)
        for l, c in zip(labels, counts):
            if c <= self.e:
                mask[self.ol == l] = False
        self.O = self.O[mask]
        self.ol = self.ol[mask]

    def _relabel_clusters(self):
        """
        Relabel clusters consecutively from 0 to n_clusters-1.
        """
        unique = np.unique(self.ol)
        mapping = {old: new for new, old in enumerate(unique)}
        self.ol = np.array([mapping[l] for l in self.ol])

