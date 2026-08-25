"""
(c) Dask backend — distributed emulation (LocalCluster, multiple workers).

Tests: does prediction actually get scheduled across several worker
processes (emulating a real cluster), not just threads in one process?

    python examples/c_dask_distributed.py
"""
import dask.array as da
from dask.distributed import Client, LocalCluster, wait
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sdoclust_parallel import SDOclust


def main():
    X, y_true = make_blobs(n_samples=200_000, centers=6, random_state=0)
    Xd = da.from_array(X, chunks=(5_000, X.shape[1]))

    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    print(f"emulated cluster: {len(client.scheduler_info()['workers'])} worker processes")

    model = SDOclust(backend="dask", zeta=0.6, chunksize=5_000, rseed=0)
    model.fit_from_dask(Xd, n_sample=20_000)

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


if __name__ == "__main__":
    main()

# capability tested: the same predict() call is actually split and executed
# by multiple independent worker processes, not just parallel threads.
