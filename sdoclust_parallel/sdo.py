import numpy as np
from scipy.spatial import distance
from sklearn.utils.validation import check_array
from .clust import ConnectedComponentsClustering

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

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
    float   - Estimated standard deviation (min 1.0).
    """
    m = X.shape[0]
    n_sample = min(m, max_samples)

    rng = np.random.default_rng(random_state)
    idx = rng.choice(m, size=n_sample, replace=False)
    Xs = X[idx]

    Xs = StandardScaler().fit_transform(Xs)
    Xp = PCA(n_components=2, random_state=random_state).fit_transform(Xs)

    return max(np.std(Xp), 1.0)

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
        rng = np.random.RandomState(random_state)
        idx = rng.choice(m, size=min(n, m), replace=False)
        return X[idx]
    import dask.array as da
    import dask.dataframe as dd
    if isinstance(X, (da.Array, dd.DataFrame)):
        if isinstance(X, da.Array):
            X = X.to_dask_dataframe()
        m = len(X)
        n_sample = min(n, m)
        frac = n_sample / m
        sampled_df = X.sample(frac=frac, random_state=random_state)
        return sampled_df.compute().to_numpy()
    else:
        raise TypeError("X must be NumPy array, Dask array, or Dask DataFrame")


def safe_knn(X, O, k, batch_size=50000):
    """
    Compute k-nearest neighbors of X with respect to O in batches.

    Parameters
    ----------
    X : ndarray, shape (n_samples_X, n_features)    - Data points to query.
    O : ndarray, shape (n_observers, n_features)    - Observer model.
    k : int                                         - Number of neighbors to retrieve.
    batch_size : int, optional                      - Rows per batch for memory efficiency.

    Returns
    -------
    dist : ndarray, shape (n_samples_X, k)          - Distances to k-nearest neighbors.
    idx : ndarray, shape (n_samples_X, k)           - Indices of k-nearest neighbors in O.
    """
    X = np.atleast_2d(X)
    n = X.shape[0]
    k = min(k, O.shape[0])

    all_dist = []
    all_idx = []

    for i in range(0, n, batch_size):
        Xi = X[i:i+batch_size]
        D = distance.cdist(Xi, O)

        idx = np.argpartition(D, k-1, axis=1)[:, :k]
        dist = np.take_along_axis(D, idx, axis=1)

        all_idx.append(idx)
        all_dist.append(dist)

    return np.vstack(all_dist), np.vstack(all_idx)


def _extend_labels_block(X_block, O, ol, knn, n_clusters, method="brute", index=None, return_labels=False):
    """
    Extend cluster labels for a block of data using the nearest observers.

    Parameters
    ----------
    X_block : ndarray, shape (n_samples_block, n_features)  - Block of data points.
    O : ndarray, shape (n_observers, n_features)            - Observer model.
    ol : ndarray, shape (n_observers,)                      - Labels of observers.
    knn : int                                               - Number of nearest observers to consider.
    n_clusters : int                                        - Total number of clusters.
    method : str, optional                                  - "brute", "faiss", or "pynndescent".
    index : object, optional                                - Precomputed index if applicable.
    return_labels : bool, optional                          - Whether to return hard labels.

    Returns
    -------
    C_norm : ndarray, shape (n_samples_block, n_clusters)   - Normalized cluster counts.
    labels : ndarray, shape (n_samples_block,), optional    - Hard labels if return_labels=True.
    """
    X_block = np.atleast_2d(X_block)
    n_obs = len(ol)
    knn_eff = min(knn, n_obs)

    if method == "brute":
        dist, closest = safe_knn(X_block, O, knn_eff)
    elif method == "faiss":
        _, closest = index.search(X_block.astype(np.float32), knn_eff)
    else:  # pynndescent
        closest, _ = index.query(X_block, k=knn_eff)

    lknn = ol[closest] 

    C = np.zeros((X_block.shape[0], n_clusters), dtype=np.float32)
    for j in range(n_clusters):
        C[:, j] = np.sum(lknn == j, axis=1)
        
    C_norm = C / knn_eff

    if return_labels:
        labels = np.argmax(C, axis=1)
        return C_norm, labels

    return C_norm

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
        dist, _ = safe_knn(X_block, O, x)
        scores = np.median(dist, axis=1)
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
        _, closest = safe_knn(X_block, O, x)
    elif method == "faiss":
        _, closest = index.search(X_block.astype(np.float32), x)
    else:
        closest, _ = index.query(X_block, k=x)
    return np.bincount(closest.ravel(), minlength=k)


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
        if self.k is None:
            sigma = estimate_sigma_pca( X,  max_samples=self.max_samples_pca,  random_state=self.rseed )
            error = 0.1 * sigma
            self.k = sample_size(m, sigma, error)
        k = self.k

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
        if self.k is None:
            sigma = estimate_sigma_pca(X, max_samples=self.max_samples_pca, random_state=self.rseed)
            error = 0.1 * sigma
            self.k = sample_size(m, sigma, error)

        O0 = random_sample(X, self.k, self.rseed)

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

    def update(self, X, cons_factor=0.1):
        """
        Update the model with new data.
        Param.:      X : array-like, shape (n_samples, n_features)   - New input data.
                     cons_factor : float, optional                   - Fraction of previous observers to retain.
        Returns:     self : SDOclust instance
        """
        self._impl.update(X, cons_factor=cons_factor)
        return self

    def update_from_dask(self, X, cons_factor=0.1, n_sample=None):
        """
        Update the model from a Dask dataset.
        Param.:      X : dask array/dataframe           - New input data.
                     cons_factor : float, optional      - Fraction of previous observers to retain.
                     n_sample : int, optional           - Number of samples to use.
        Returns:     self : SDOclust instance
        """
        return self._impl.update_from_dask(X, cons_factor=cons_factor, n_sample=n_sample)

    def update_predict(self, X, return_membership=False, xc=None, cons_factor=0.1):
        """
        Update the model and predict cluster labels.
        Param.:      X : array-like, shape (n_samples, n_features)   - New input data.
                     return_membership : bool, optional              - If True, also return membership degrees.
                     xc : int, optional                              - Number of observers to use for extending labels.
                     cons_factor : float, optional                   - Fraction of previous observers to retain.
        Returns:     labels : ndarray, shape (n_samples,)            - Cluster labels for each sample, or
                    (membership, labels) : tuple of ndarrays         - If return_membership=True.
        """
        return self._impl.update_predict(X, return_membership=return_membership, xc=xc, cons_factor=cons_factor)

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

    def fit(self, X):
        X = check_array(X, accept_sparse="csr")

        self.sdo = SDO_numpy( x=self.x, qv=self.qv, chunksize=self.chunksize, method=self.method,  max_samples_pca=self.max_samples_pca, k=self.k,            rseed=self.rseed, n_jobs=self.n_jobs).fit(X)

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
        if self.n_jobs is None or self.n_jobs == 1:
            results = [_extend_labels_block(b, self.O, self.ol, xc, self.n_clusters, self.method, self.index, True) for b in blocks]
        else:
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_extend_labels_block)(b, self.O, self.ol, xc, self.n_clusters, self.method, self.index, True) for b in blocks )

        membership = np.vstack([r[0] for r in results])
        labels = np.concatenate([r[1] for r in results])

        return (membership, labels) if return_membership else labels

    def fit_predict(self, X, return_membership=False, xc=None):
        self.fit(X)
        return self.predict(X, return_membership=return_membership, xc=xc)

    def update(self, X, cons_factor=0.1):
        X = check_array(X, accept_sparse="csr")

        m = X.shape[0]
        k_old = self.O.shape[0]
        k_new = max(1, int(m * self.kmp))
        k_tot = k_old + k_new

        O_new = random_sample(X, k_new, self.rseed)
        O_ext = np.vstack((self.O, O_new))

        index = self._build_index_for(O_ext)

        blocks = [X[i:i+self.chunksize] for i in range(0, m, self.chunksize)]
        if self.n_jobs is None or self.n_jobs == 1:
            P = np.sum([_observer_count_block(b, O_ext, self.xc, self.method, index) for b in blocks], axis=0)
        else:
            counts = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_observer_count_block)(b, O_ext, self.xc, self.method, index) for b in blocks )
            P = np.sum(counts, axis=0)

        n_cons = int(len(self.O) * cons_factor)
        keep_cons_idx = np.random.choice(len(self.O), size=n_cons, replace=False)
        mask_candidates = np.ones(len(O_ext), dtype=bool)
        mask_candidates[keep_cons_idx] = False
        toremove = np.argsort(P[mask_candidates])[:k_new]
        candidates_idx = np.where(mask_candidates)[0]
        toremove = candidates_idx[toremove]
        keep = np.ones(len(O_ext), dtype=bool)
        keep[toremove] = False
        self.O = O_ext[keep]

        self.ol = ConnectedComponentsClustering(zeta=self.zeta, chi=self.chi, chi_min=self.chi_min, chi_prop=self.chi_prop, n_jobs=self.n_jobs).fit(self.O).labels_

        self._relabel_clusters()
        self._build_index()

        self.kmp = self.O.shape[0] / m
        self.n_clusters = len(np.unique(self.ol))

        return self

    def update_predict(self, X, return_membership=False, xc=None, cons_factor=0.1):
        self.update(X, cons_factor=cons_factor)
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

    def update_from_dask(self, X, n_sample=None, cons_factor=0.1):
        if n_sample is None:
            n_sample = self.chunksize
        Xs = random_sample(X, n_sample, self.rseed)
        return self.update(Xs, cons_factor=cons_factor)

    def fit(self, X):
        X = check_array(X, accept_sparse="csr")

        self.sdo = SDO_dask(
            x=self.x, qv=self.qv, chunksize=self.chunksize, k=self.k,
            rseed=self.rseed, max_samples_pca=self.max_samples_pca, n_jobs=self.n_jobs
        ).fit(X)

        self.O = self.sdo.O

        self.ol = ConnectedComponentsClustering(
            zeta=self.zeta, chi=self.chi, chi_min=self.chi_min, chi_prop=self.chi_prop,
            n_jobs=self.n_jobs
        ).fit(self.O).labels_

        self._remove_small_clusters()
        self._relabel_clusters()
        self.n_clusters = len(np.unique(self.ol))

        return self

    def update(self, X, cons_factor=0.1):
        X = check_array(X, accept_sparse="csr")
        m = X.shape[0]

        k_new = max(1, int(m * (self.O.shape[0] / m)))
        O_new = random_sample(X, k_new, self.rseed)

        O_ext = np.vstack((self.O, O_new))

        blocks = [X[i:i+self.chunksize] for i in range(0, m, self.chunksize)]
        if self.n_jobs is None or self.n_jobs == 1:
            P = np.sum([_observer_count_block(b, O_ext, self.xc, "brute", None) for b in blocks], axis=0)
        else:
            counts = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_observer_count_block)(b, O_ext, self.xc, "brute", None) for b in blocks )
            P = np.sum(counts, axis=0)

        n_cons = int(len(self.O) * cons_factor)
        keep_cons_idx = np.random.choice(len(self.O), size=n_cons, replace=False)
        mask_candidates = np.ones(len(O_ext), dtype=bool)
        mask_candidates[keep_cons_idx] = False
        toremove = np.argsort(P[mask_candidates])[:k_new]
        candidates_idx = np.where(mask_candidates)[0]
        toremove = candidates_idx[toremove]
        keep = np.ones(len(O_ext), dtype=bool)
        keep[toremove] = False
        self.O = O_ext[keep]

        self.ol = ConnectedComponentsClustering( zeta=self.zeta, chi=self.chi, chi_min=self.chi_min, chi_prop=self.chi_prop,
            n_jobs=self.n_jobs).fit(self.O).labels_

        self._relabel_clusters()
        self.n_clusters = len(np.unique(self.ol))
        self.sdo.O = self.O
        self.kmp = self.O.shape[0] / m

        return self


    def predict(self, X, return_membership=False, xc=None):
        xc = xc if xc is not None else self.xc
        Xd = self._as_dask_array(X)
        n_clusters = self.n_clusters

        # map function for blocks
        def _map_labels(block, O, ol, knn, n_clusters):
            return _extend_labels_block(block, O, ol, knn=knn, n_clusters=n_clusters, method="brute")

        Md = Xd.map_blocks(
            _map_labels, O=self.O, ol=self.ol, knn=xc, n_clusters=n_clusters,
            dtype=float, chunks=(Xd.chunks[0], n_clusters)
        )

        labels = self._da.argmax(Md, axis=1)

        if return_membership:
            return Md, labels
        else:
            return labels

    def fit_predict(self, X, return_membership=False, xc=None):
        if isinstance(X, self._da.Array):
            self.fit_from_dask(X)
        else:
            self.fit(X)
        return self.predict(X, return_membership=return_membership, xc=xc)

    def update_predict(self, X, return_membership=False, xc=None, cons_factor=0.1):
        self.update_from_dask(X, cons_factor=cons_factor)
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

