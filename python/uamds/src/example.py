import numpy as np
import matplotlib.pyplot as plt
import scipy
import pokemon
import uamds
import util
import uapca


def main():
    pokemon_distribs = pokemon.get_normal_distribs()
    # get first 9 pokemon (1st gen starters)
    n = 9
    distrib_set = pokemon_distribs[0:n]
    types = pokemon.get_type1()[0:n]
    means_hi = [d.mean for d in distrib_set]
    covs_hi = [d.cov for d in distrib_set]

    # prepare data matrix consisting of a block of means followed by a block of covs
    distrib_spec = util.mk_normal_distr_spec(means_hi, covs_hi)
    # compute UAPCA projection
    uapca_means, uapca_covs = uapca.transform_uapca(distrib_spec[0:n, :], distrib_spec[n:, :])
    uapca_means, uapca_covs = util.get_means_covs(np.vstack([uapca_means, uapca_covs]))
    plot_2d_normal_distribs(uapca_means, uapca_covs, types, pokemon.get_type_colors())

    hi_d = distrib_spec.shape[1]
    lo_d = 2
    pre = uamds.precalculate_constants(distrib_spec)
    # random initialization
    uamds_transforms = np.random.rand(distrib_spec.shape[0], lo_d)
    # test plausibility of gradient implemenentation against stress
    check_gradient(distrib_spec, uamds_transforms, pre)
    # perform UAMDS
    uamds_transforms = uamds.iterate_simple_gradient_descent(
        distrib_spec, uamds_transforms, pre, num_iter=1000, a=0.0001)
    # project distributions
    distrib_spec_lo = uamds.perform_projection(distrib_spec, uamds_transforms)
    means_lo, covs_lo = util.get_means_covs(distrib_spec_lo)
    plot_2d_normal_distribs(means_lo, covs_lo, types, pokemon.get_type_colors())



def check_gradient(distrib_spec: np.ndarray, uamds_transforms: np.ndarray, pre: tuple):
    x_shape = uamds_transforms.shape
    n_elems = uamds_transforms.size

    def fx(x: np.ndarray):
        return uamds.stress(distrib_spec, x.reshape(x_shape), pre)

    def dfx(x: np.ndarray):
        return uamds.gradient(distrib_spec, x.reshape(x_shape), pre).reshape(n_elems)

    err = scipy.optimize.check_grad(fx, dfx, uamds_transforms.reshape(uamds_transforms.size))
    print(f"gradient approximation error: {err}")


def plot_2d_normal_distribs(means: list[np.ndarray], covs: list[np.ndarray], labels, colormap):
    n = len(means)
    # draw samples from 2D normal distributions
    n_samples = 100
    samples = np.vstack([np.random.multivariate_normal(means[i], covs[i], size=n_samples) for i in range(n)])
    sample_labels = [labels[j] for j in range(n) for i in range(n_samples)]
    sample_colors = [colormap[label] for label in sample_labels]
    # make vis
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('equal')
    ax.scatter(samples[:, 0], samples[:, 1], c=sample_colors, s=2)
    # confidence ellipses for each distribution
    for i in range(n):
        util.confidence_ellipse(means[i], covs[i], ax, n_std=1, edgecolor=colormap[labels[i]])
        util.confidence_ellipse(means[i], covs[i], ax, n_std=2, edgecolor=colormap[labels[i]])
        util.confidence_ellipse(means[i], covs[i], ax, n_std=3, edgecolor=colormap[labels[i]])

    plt.show()



if __name__ == '__main__':
    main()




