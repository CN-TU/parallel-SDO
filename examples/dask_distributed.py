"""
(c) Dask backend — distributed emulation (LocalCluster, multiple workers).
"""

import dask.array as da
import numpy as np
from dask.distributed import Client, LocalCluster, wait
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sdoclust_parallel import SDOclust


def main():
    X, y_true = make_blobs(n_samples=200000, centers=6, cluster_std=0.5, random_state=0)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} dimensions, {len(np.unique(y_true))} clusters")
    Xd = da.from_array(X, chunks=(5000, X.shape[1]))

    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    print(f"emulated cluster: {len(client.scheduler_info()['workers'])} worker processes")

    model = SDOclust(backend="dask", chunksize=5000, rseed=0)
    model.fit_from_dask(Xd, n_sample=20000)

    labels_lazy = model.predict(Xd).persist()  # schedules work on the cluster now
    wait(labels_lazy)                          # block until all tasks finish
    who = client.who_has(labels_lazy)
    per_worker = {}
    for _, workers in who.items():
        for w in workers:
            per_worker[w] = per_worker.get(w, 0) + 1
    print(f"tasks distributed across workers: {per_worker}")

    labels = labels_lazy.compute()
    ari = adjusted_rand_score(y_true, labels)
    print(f"quality(ARI)={ari:.3f}")

    client.close()
    cluster.close()

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=1, alpha=0.5, cmap='viridis')
    plt.colorbar(label='predictions')
    plt.xlabel("feat. 1")
    plt.ylabel("feat. 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()


