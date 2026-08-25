from typing import Callable, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import validate_data
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class ConnectedComponentsClustering(ClusterMixin, BaseEstimator):
    def __init__( self, zeta: float, chi: Optional[int], chi_min: int, chi_prop: float, metric: Union[str, Callable] = "euclidean", n_jobs: Optional[int] = None,
    ) -> None:
        self.zeta = zeta
        self.chi = chi
        self.chi_min = chi_min
        self.chi_prop = chi_prop
        self.metric = metric
        self.n_jobs = n_jobs
        self.threshold_ = None

    def fit(self, X: np.ndarray):
        X = validate_data(self, X, accept_sparse="csr")

        # Memory and speed optimization
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)

        # calculates chi and searches chi+1 closest neighbours
        n_samples = X.shape[0]
        chi = self.chi or max(int(n_samples * self.chi_prop), self.chi_min) 
        nn = NearestNeighbors( n_neighbors=chi + 1, metric=self.metric, n_jobs=self.n_jobs, ) 
        distances, indices = nn.fit(X).kneighbors(X)
        
        # calculates threshold for cutting sub-graphs
        aux = distances[:, chi]
        mean_aux = aux.mean()
        threshold = self.zeta * aux + (1.0 - self.zeta) * mean_aux
        self.threshold_ = threshold

        # building the adjacency matrix based on threshold
        rows, cols = [], []
        for i in range(n_samples):
            mask = distances[i, 1:] < threshold[i]
            neighs = indices[i, 1:][mask]
            rows.extend([i] * len(neighs))
            cols.extend(neighs)
        adjacency = csr_matrix( (np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n_samples, n_samples), )
        adjacency = adjacency.minimum(adjacency.T) # makes it symmetrical
        
        # assigns labels to connected components
        _, labels = connected_components(csgraph=adjacency, directed=False, return_labels=True )
        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).labels_

