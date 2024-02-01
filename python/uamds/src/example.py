import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import pokemon
import uamds
import util
import uapca
import time


def main1():
    pokemon_distribs = pokemon.get_normal_distribs()
    # get first 9 pokemon (1st gen starters)
    n = 15
    distrib_set = pokemon_distribs[0:n]
    types = pokemon.get_type1()[0:n]
    means_hi = [d.mean for d in distrib_set]
    covs_hi = [d.cov for d in distrib_set]
    # prepare data matrix consisting of a block of means followed by a block of covs
    distrib_spec = util.mk_normal_distr_spec(means_hi, covs_hi)
    # dimensionality and precomputation for UAMDS
    hi_d = distrib_spec.shape[1]
    lo_d = 2
    pre = uamds.precalculate_constants(distrib_spec)
    # random initialization
    np.random.seed(0)
    uamds_transforms = np.random.rand(distrib_spec.shape[0], lo_d)
    # test plausibility of gradient implemenentation against stress
    check_gradient(distrib_spec, uamds_transforms, pre)
    # perform UAMDS over and over again
    n_repetitions = 10000
    matplotlib.use("TkAgg")  # may require on linux: sudo apt-get install python3-tk
    with plt.ion():
        fig = ax = None
        for rep in range(n_repetitions):
            start = time.time_ns()
            #uamds_transforms = uamds.iterate_simple_gradient_descent(distrib_spec, uamds_transforms, pre, num_iter=1000, a=0.002)
            uamds_transforms = uamds.iterate_scipy(distrib_spec, uamds_transforms, pre)
            stop = time.time_ns()
            print(f"stress: {uamds.stress(distrib_spec, uamds_transforms, pre)} in {(stop-start)/1000_000_000}s")
            # project distributions
            distrib_spec_lo = uamds.perform_projection(distrib_spec, uamds_transforms)
            means_lo, covs_lo = util.get_means_covs(distrib_spec_lo)
            if ax is not None:
                ax.clear()
            fig, ax = plot_normal_distrib_contours(means_lo, covs_lo, types, pokemon.get_type_colors(), fig=fig, ax=ax)
            plt.pause(0.1)

    print("done")


def main2():
    pokemon_distribs = pokemon.get_normal_distribs()
    # get first 9 pokemon (1st gen starters)
    n = 9
    distrib_set = pokemon_distribs[0:n]
    types = pokemon.get_type1()[0:n]
    means_hi = [d.mean for d in distrib_set]
    covs_hi = [d.cov for d in distrib_set]
    # prepare data matrix consisting of a block of means followed by a block of covs
    distrib_spec = util.mk_normal_distr_spec(means_hi, covs_hi)
    hi_d = distrib_spec.shape[1]
    lo_d = 2
    # compute UAPCA projection
    eigvecs, eigvals = uapca.compute_uapca(distrib_spec[0:n, :], distrib_spec[n:, :])
    projmat = eigvecs[:,:2]
    # create initialization for UAMDS from UAPCA (each distribution has translation 0 and the same projection matrix)
    translations = np.zeros((n,2))
    projection_mats = np.vstack([projmat for i in range(n)])
    affine_transforms = np.vstack([translations, projection_mats])
    uamds_transforms = uamds.convert_xform_affine_to_uamds(distrib_spec, affine_transforms)
    # perform UAMDS
    pre = uamds.precalculate_constants(distrib_spec)
    uamds_transforms = uamds.iterate_simple_gradient_descent(
        distrib_spec, uamds_transforms, pre, num_iter=1000, a=0.0001)
    # project high dimensional samples
    n_samples = 500
    samples = [np.random.multivariate_normal(distrib_set[i].mean, distrib_set[i].cov, size=n_samples) for i in range(n)]
    affine_transforms = uamds.convert_xform_uamds_to_affine(distrib_spec, uamds_transforms)
    translations = affine_transforms[:n,:]
    projection_mats = affine_transforms[n:,:]
    projected_samples = [samples[i] @ projection_mats[i*hi_d:(i+1)*hi_d,:] + translations[i,:] for i in range(n)]
    # show samples
    labels = types
    colormap = pokemon.get_type_colors()
    sample_labels = [labels[j] for j in range(n) for i in range(n_samples)]
    sample_colors = [colormap[label] for label in sample_labels]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('equal')
    ax.scatter(np.vstack(projected_samples)[:, 0], np.vstack(projected_samples)[:, 1], c=sample_colors, s=2)
    plt.show()
    


def check_gradient(distrib_spec: np.ndarray, uamds_transforms: np.ndarray, pre: tuple):
    x_shape = uamds_transforms.shape
    n_elems = uamds_transforms.size

    def fx(x: np.ndarray):
        return uamds.stress(distrib_spec, x.reshape(x_shape), pre)

    def dfx(x: np.ndarray):
        return uamds.gradient(distrib_spec, x.reshape(x_shape), pre).reshape(n_elems)

    err = scipy.optimize.check_grad(fx, dfx, uamds_transforms.reshape(uamds_transforms.size))
    print(f"gradient approximation error: {err}")


def plot_normal_distrib_contours(means: list[np.ndarray], covs: list[np.ndarray], labels, colormap, fig=None, ax=None):
    n = len(means)
    # make vis
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('equal')
    colors = [colormap[label] for label in labels]
    ax.scatter(np.vstack(means)[:,0], np.vstack(means)[:,1], c=colors, s=2)
    # confidence ellipses for each distribution
    for i in range(n):
        util.confidence_ellipse(means[i], covs[i], ax, n_std=1, edgecolor=colormap[labels[i]])
        util.confidence_ellipse(means[i], covs[i], ax, n_std=2, edgecolor=colormap[labels[i]])
        util.confidence_ellipse(means[i], covs[i], ax, n_std=3, edgecolor=colormap[labels[i]])
    return fig, ax


def plot_normal_distrib_samples(means: list[np.ndarray], covs: list[np.ndarray], labels, colormap, n_samples: int=100):
    n = len(means)
    # draw samples from 2D normal distributions
    samples = np.vstack([np.random.multivariate_normal(means[i], covs[i], size=n_samples) for i in range(n)])
    sample_labels = [labels[j] for j in range(n) for i in range(n_samples)]
    sample_colors = [colormap[label] for label in sample_labels]
    # make vis
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('equal')
    ax.scatter(samples[:, 0], samples[:, 1], c=sample_colors, s=2)



if __name__ == '__main__':
    main1()




