import numpy as np


def compute_ua_cov(means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    n = means.shape[0]
    d = means.shape[1]
    # empirical mean
    mu = means.mean(axis=0)
    # centering matrix
    centering = np.outer(mu, mu)
    # average covariance matrix
    avg_cov = covs.reshape((n,d,d)).mean(axis=0)
    # sample covariance
    sample_cov = np.array([np.outer(means[i,:], means[i,:]) for i in range(n)]).mean(axis=0)
    # final uncertainty aware covariance matrix
    return sample_cov + avg_cov - centering


def compute_uapca(means: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = compute_ua_cov(means, covs)
    u,s,vh = np.linalg.svd(cov, full_matrices=True)
    return u, s


def transform_uapca(means, covs, dims: int=2) -> tuple[np.ndarray, np.ndarray]:
    n = means.shape[0]
    d = means.shape[1]
    eigvecs, eigvals = compute_uapca(means, covs)
    projmat = eigvecs[:, :dims]
    projected_means = means @ projmat
    projected_covs = np.vstack(
        [projmat.T @ covs[i*d:(i+1)*d, :] @ projmat for i in range(n)]
    )
    return projected_means, projected_covs
