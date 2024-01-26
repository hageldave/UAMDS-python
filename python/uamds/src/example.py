import numpy as np
import matplotlib.pyplot as plt
import pokemon
import uamds
import util


def main():
    pokemon_distribs = pokemon.get_normal_distribs()
    # get first 9 pokemon (1st gen starters)
    n = 9
    distrib_set = pokemon_distribs[0:n]
    types = pokemon.get_type1()[0:n]
    means_hi = [d.mean for d in distrib_set]
    covs_hi = [d.cov for d in distrib_set]

    # prepare for uamds
    distrib_spec = uamds.mk_normal_distr_spec(means_hi, covs_hi)
    hi_d = distrib_spec.shape[1]
    lo_d = 2
    affine_transforms = np.random.rand(distrib_spec.shape[0], lo_d)
    pre = uamds.precalculate_constants(distrib_spec)
    # perform UAMDS
    affine_transforms = uamds.iterate_simple_gradient_descent(
        distrib_spec, affine_transforms, pre, num_iter=20000, a=0.0001)
    # project distributions
    distrib_spec_lo = uamds.perform_projection(distrib_spec, affine_transforms)
    means_lo, covs_lo = uamds.get_means_covs(distrib_spec_lo)
    plot_2d_normal_distribs(means_lo, covs_lo, types, pokemon.get_type_colors())


def plot_2d_normal_distribs(means: list[np.ndarray], covs: list[np.ndarray], labels, colormap):
    n = len(means)
    # draw samples from 2D normal distributions
    n_samples = 100
    samples = np.vstack([np.random.multivariate_normal(means[i], covs[i], size=n_samples) for i in range(n)])
    sample_labels = [labels[j] for j in range(n) for i in range(n_samples)]
    sample_colors = [colormap[label] for label in sample_labels]
    # make vis
    fig, ax = plt.subplots()  #figsize=(6, 6))
    ax.scatter(samples[:, 0], samples[:, 1], c=sample_colors, s=2)
    # confidence ellipses for each distribution
    for i in range(n):
        util.confidence_ellipse(means[i], covs[i], ax, n_std=1, edgecolor=colormap[labels[i]])
        util.confidence_ellipse(means[i], covs[i], ax, n_std=2, edgecolor=colormap[labels[i]])
        util.confidence_ellipse(means[i], covs[i], ax, n_std=3, edgecolor=colormap[labels[i]])

    plt.show()



if __name__ == '__main__':
    main()




