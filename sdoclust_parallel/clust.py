import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data


def distances_to_adjacency_matrix(distances, indices, threshold):
    N = distances.shape[0]

    mask = distances[:, 1:] < threshold[:, None]

    rows = np.repeat(np.arange(N), mask.sum(axis=1))
    cols = indices[:, 1:][mask]

    adjacency_matrix = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(N, N))
    return adjacency_matrix.minimum(adjacency_matrix.T)


class ConnectedComponentsClustering(ClusterMixin, BaseEstimator):

    def __init__(self, zeta, chi, chi_min, chi_prop, metric="euclidean", n_jobs=None):
        self.zeta = zeta
        self.chi = chi
        self.chi_min = chi_min
        self.chi_prop = chi_prop
        self.metric = metric
        self.n_jobs = n_jobs
        self.threshold_ = None

    def fit(self, X: np.ndarray):
        X = validate_data(self, X, accept_sparse="csr")

        # Optimizacion de memoria y velocidad
        X = X.astype(np.float32, copy=False)

        m = X.shape[0]
        chi = self.chi or max(int(m * self.chi_prop), self.chi_min)

        # Se buscan los chi + 1 vecinos mas cercanos (el primero es el propio punto)
        nn = NearestNeighbors(n_neighbors=chi + 1, metric=self.metric, n_jobs=self.n_jobs)
        distances, indices = nn.fit(X).kneighbors(X)

        # Umbral de corte por punto
        aux = distances[:, chi]
        mean_aux = np.mean(aux)
        self.threshold_ = self.zeta * aux + (1 - self.zeta) * mean_aux

        adjacency_matrix = distances_to_adjacency_matrix(distances, indices, self.threshold_)

        labels = connected_components(csgraph=adjacency_matrix, directed=False, return_labels=True)[1]
        self.labels_ = labels

        return self

    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.labels_
