import numpy as np
import math


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
