import numpy as np
import math
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def detect_rotation(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    # centering points
    points_a -= points_a.mean(axis=0)
    points_b -= points_b.mean(axis=0)

    # ------------------------------------------------------------------------------------------------------------------
    # assuming points_b is a rotated version of points_a
    #                   cos(a) sin(a)
    # (pa.x  pa.y)  *  -sin(a) cos(a)  =  (pb.x  pb.y)
    #
    # -->  pa.x * cos - pa.y * sin = pb.x
    # -->  pa.x * sin + pa.y * cos = pb.y
    #
    # now we want to find an angle so that rotating all pa by that angle will give us points that are closest to pb
    #
    # minimize SUM_i ((pa[i].x * cos - pa[i].y * sin) - pb[i].x)^2 + ((pa[i].x * sin + pa[i].y * cos) - pb[i].y)^2
    #
    # let's rename variables pa[i].x=a, pa[i].y=b, pb[i].x=c, pb[i].y=d, a=x
    # so we have the term ((a*cos(x) - b*sin(x))-c)^2 + ((a*sin(x) + b*cos(x))-d)^2    (omitting SUM_i for simplicity)
    #
    # taking derivative wrt. x  -->  2 ((b*c - a*d)*cos(x) + (a*c + b*d)*sin(x)) = 0
    #              simplifying  -->  u * cos(x) + v * sin(x) = 0
    #              solution     -->  x = PI*n - atan(u/v)  with integer n
    #
    #          re-substitution  -->  u = (b*c - a*d) = SUM_i (pa[i].y*pb[i].x - pa[i].x*pb[i].y)
    #                                v = (a*c + b*d) = SUM_i (pa[i].x*pb[i].x - pa[i].y*pb[i].y)
    # ------------------------------------------------------------------------------------------------------------------
    a = points_a[:, 0]
    b = points_a[:, 1]
    c = points_b[:, 0]
    d = points_b[:, 1]

    u = ((b*c) - (a*d)).sum()
    v = ((a*c) + (b*d)).sum()

    angle = math.atan(u/v)  # ignoring larger possible angles resulting from +PI*n
    print(angle)
    # now we obtain the rotation matrix that undoes the rotation of points_b
    rot_reverse = np.array([[math.cos(-angle), -math.sin(-angle)], [math.sin(-angle), math.cos(-angle)]])
    return rot_reverse


def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    mean : np.ndarray of shape (2,)

    cov : np.ndarray of shape (2,2)

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    color : Any
        line color of the ellipse

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# def get_means_covs(normal_distr_spec: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
#     d_hi = normal_distr_spec.shape[1]
#     n = normal_distr_spec.shape[0] // (d_hi + 1)
#     means = []
#     covs = []
#     for i in range(n):
#         means.append(normal_distr_spec[i, :])
#         covs.append(normal_distr_spec[n+i*d_hi : n+(i+1)*d_hi, :])
#     return means, covs
#
#
# def mk_normal_distr_spec(means: list[np.ndarray], covs: list[np.ndarray]) -> np.ndarray:
#     mean_block = np.vstack(means)
#     cov_block = np.vstack(covs)
#     return np.vstack([mean_block, cov_block])
