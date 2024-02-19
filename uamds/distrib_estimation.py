import numpy as np
from typing import Any, Callable


class MVN:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self, n: int) -> np.ndarray:
        return np.random.multivariate_normal(self.mean, self.cov, size=n)

    def __str__(self):
        return f"MVN(mean={self.mean}, cov=\n{self.cov})\n"

    def __repr__(self):
        return self.__str__()


def estimate_normal_distribs(
        data: np.ndarray,
        labels
) -> dict[Any, MVN]:
    unique_labels = set()
    for label in labels:
        unique_labels.add(label)

    distribs = {}
    for label in unique_labels:
        dat = np.array([data[i] for i in range(data.shape[0]) if labels[i] == label])
        mean = dat.mean(axis=0)
        dat = dat-mean
        cov = (dat.T @ dat) * (1.0/dat.shape[0])

        distribs[label] = MVN(mean, cov)
    return distribs


