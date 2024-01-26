import numpy as np


def precalculate_constants(normal_distr_spec: np.ndarray) -> dict:
    d_hi = normal_distr_spec.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi+1)  # array of (d_hi x d_hi) cov matrices and (1 x d_hi) means

    # extract means and covs
    mu = [normal_distr_spec[i, :] for i in range(n)]
    cov = [normal_distr_spec[n+d_hi*i:n+d_hi*(i+1), :] for i in range(n)]

    # compute singular value decomps of covs
    svds = [np.linalg.svd(cov[i], full_matrices=True) for i in range(n)]
    U = [svds[i].U for i in range(n)]
    S = [np.diag(svds[i].S) for i in range(n)]
    Ssqrt = [np.diag(np.sqrt(svds[i].S)) for i in range(n)]

    # combinations used in stress terms
    norm2_mui_sub_muj = [[np.dot(mu[i]-mu[j], mu[i]-mu[j]) for j in range(n)] for i in range(n)]
    Ssqrti_UiTUj_Ssqrtj = [[Ssqrt[i] @ U[i].T @ U[j] @ Ssqrt[j] for j in range(n)] for i in range(n)]
    mui_sub_muj_TUi = [[(mu[i]-mu[j]) @ U[i] for j in range(n)] for i in range(n)]
    mui_sub_muj_TUj = [[(mu[i]-mu[j]) @ U[j] for j in range(n)] for i in range(n)]
    Zij = [[U[i].T @ U[j] for j in range(n)] for i in range(n)]

    constants = {
        'mu': mu,
        'cov': cov,
        'U': U,
        'S': S,
        'Ssqrt': Ssqrt,
        'norm2_mui_sub_muj': norm2_mui_sub_muj,
        'Ssqrti_UiTUj_Ssqrtj': Ssqrti_UiTUj_Ssqrtj,
        'mui_sub_muj_TUi': mui_sub_muj_TUi,
        'mui_sub_muj_TUj': mui_sub_muj_TUj,
        'Zij': Zij
    }
    return constants


def stress(normal_distr_spec: np.ndarray, affine_transforms: np.ndarray, precalc_constants: dict=None) -> float:
    d_hi = normal_distr_spec.shape[1]
    d_lo = affine_transforms.shape[1]
    n = normal_distr_spec.shape[0]//(d_hi+1)  # array of (d_hi x d_hi) cov matrices and (1 x d_hi) means

    if precalc_constants is None:
        precalc_constants = precalculate_constants(normal_distr_spec)

def stress_ij(i: int, j: int, normal_distr_spec: np.ndarray, affine_transforms: np.ndarray, pre: dict) -> float:
    d_hi = normal_distr_spec.shape[1]
    d_lo = affine_transforms.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi + 1)
    # get some objects for i
    Si = pre['S'][i]
    Ssqrti = pre['Ssqrt'][i]
    ci = affine_transforms[i, :]
    Bi = affine_transforms[n+i*d_hi : n+(i+1)*d_hi, :]

    # get some objects for j
    Sj = pre['S'][j]
    Ssqrtj = pre['Ssqrt'][j]
    cj = affine_transforms[j, :]
    Bj = affine_transforms[n+j*d_hi : n+(j+1)*d_hi, :]

    ci_sub_cj = ci-cj

    # compute term 1 : part 1 : ||Si - Si^(1/2)Bi^T BiSi^(1/2)||_F^2
    temp = Ssqrti @ Bi
    temp = Si - (temp @ temp.T)
    part1 = (temp*temp).sum()  # sum of squared elements = squared frobenius norm
    # compute term 1 : part 2 : same as part 1 but with j
    temp = Ssqrtj @ Bj
    temp = Sj - (temp @ temp.T)
    part2 = (temp*temp).sum()  # sum of squared elements = squared frobenius norm
    # compute term 1 : part 3
    temp = (Ssqrti @ Bi) @ (Bj.T @ Ssqrtj)  # outer product of transformed Bs
    temp = pre['Ssqrti_UiTUj_Ssqrtj'][i][j] - temp
    part3 = (temp*temp).sum()  # sum of squared elements = squared frobenius norm
    term1 = 2*(part1+part2)+4*part3

    # compute term 2 : part 1 : sum_k^n [ Si_k * ( <Ui_k, mui-muj> - <Bi_k, ci-cj> )^2 ]
    temp = ci_sub_cj @ Bi.T
    temp = pre['mui_sub_muj_TUi'][i][j] - temp
    temp = temp*temp  # squared
    part1 = (temp @ Si).sum()
    # compute term 2 : part 2 : same as part 1 but with j
    temp = ci_sub_cj @ Bj.T
    temp = pre['mui_sub_muj_TUj'][i][j] - temp
    temp = temp*temp  # squared
    part2 = (temp @ Sj).sum()
    term2 = part1+part2

    # compute term 3 : part 1
    norm1 = pre['norm2_mui_sub_muj'][i][j]
    norm2 = np.dot(ci_sub_cj,ci_sub_cj)  # squared norm
    part1 = norm1-norm2
    # compute term 3 : part 2
    part2 = 0
    part3 = 0
    for k in range(d_hi):
        sigma_i = Si[k, k]
        sigma_j = Sj[k, k]
        bik = Bi[k, :]
        bjk = Bj[k, :]
        part2 += (1 - np.dot(bik,bik))*sigma_i
        part3 += (1 - np.dot(bjk,bjk))*sigma_j
    term3 = (part1 + part2 + part3)**2

    return term1+term2+term3


def gradient_ij(i: int, j: int, normal_distr_spec: np.ndarray, affine_transforms: np.ndarray, pre: dict) -> np.ndarray:
    d_hi = normal_distr_spec.shape[1]
    d_lo = affine_transforms.shape[1]
    n = normal_distr_spec.shape[0] // (d_hi + 1)
    # get some objects for i
    Si = pre['S'][i]
    mui = pre['mu'][i]
    ci = affine_transforms[i, :]
    Bi = affine_transforms[n + i * d_hi: n + (i + 1) * d_hi, :].T
    BiSi = Bi @ Si


    # get some objects for j
    Sj = pre['S'][j]
    muj = pre['mu'][j]
    cj = affine_transforms[j, :]
    Bj = affine_transforms[n + j * d_hi: n + (j + 1) * d_hi, :].T
    BjSj = Bj @ Sj

    mui_sub_muj = mui - muj
    ci_sub_cj = ci - cj

    # compute term 1 :
    Zij = pre['Zij'][i][j]
    part1i = (BiSi @ Bi.T @ BiSi) - (BiSi @ Si)
    part1j = (BjSj @ Bj.T @ BjSj) - (BjSj @ Sj)
    part2i = (BjSj @ Bj.T @ BiSi) - (BjSj @ Zij.T @ Si)
    part2j = (BiSi @ Bi.T @ BjSj) - (BiSi @ Zij   @ Sj)
    dBi = (part1i + part2i) * 8
    dBj = (part1j + part2j) * 8

    # compute term 2 :
    dci = ci*0
    dcj = cj*0
    if i != j:
        # gradient part for B matrices
        part3i = (np.outer(ci_sub_cj, (ci_sub_cj @ Bi)) - np.outer(ci_sub_cj, pre['mui_sub_muj_TUi'][i][j])) @ Si
        part3j = (np.outer(ci_sub_cj, (ci_sub_cj @ Bj)) - np.outer(ci_sub_cj, pre['mui_sub_muj_TUj'][i][j])) @ Sj
        dBi += 2*part3i
        dBj += 2*part3j
        # gradient part for c vectors
        part4i = (pre['mui_sub_muj_TUi'][i][j] - (ci_sub_cj @ Bi)) @ BiSi.T
        part4j = (pre['mui_sub_muj_TUj'][i][j] - (ci_sub_cj @ Bj)) @ BjSj.T
        part4 = -2*(part4i+part4j)
        dci += part4
        dcj -= part4

    # compute term 3 :
    norm1 = pre['norm2_mui_sub_muj'][i][j]
    norm2 = np.dot(ci_sub_cj, ci_sub_cj)
    part1 = norm1-norm2
    part2 = part3 = 0.0
    for k in range(d_hi):
        sigma_i = Si[k, k]
        sigma_j = Sj[k, k]
        bik = Bi[:, k]
        bjk = Bj[:, k]
        part2 += (1 - np.dot(bik, bik)) * sigma_i
        part3 += (1 - np.dot(bjk, bjk)) * sigma_j
    term3 = -4 * (part1 + part2 + part3)
    dBi += BiSi * term3
    dBj += BjSj * term3

    if i != j:
        dci += ci_sub_cj * term3
        dcj -= ci_sub_cj * term3

    return dBi, dBj, dci, dcj

def main():
    n = 4
    d = 6
    lo_d = 2

    means = np.vstack([np.ones(d)*(i+1) for i in range(n)])
    covs = np.vstack([mk_cov_mat(d, i+1) for i in range(n)])
    normal_distr_spec = np.vstack([means, covs])
    constants = precalculate_constants(normal_distr_spec)

    c = np.vstack([np.ones(lo_d)*(i+1) for i in range(n)])
    B = np.vstack([np.ones((d,lo_d))*(i*0.1+0.1) for i in range(n)])
    affine_transforms = np.vstack([c,B])

    s = stress_ij(1,2, normal_distr_spec, affine_transforms, constants)
    dbi, dbj, dci, dcj = gradient_ij(1,2, normal_distr_spec, affine_transforms, constants)
    print(s)
    print(dbi)
    print(dbj)
    print(dci)
    print(dcj)

def mk_cov_mat(d, s):
    a = np.array([i*s for i in range(d*d)])
    a = a.reshape((d,d))
    a = np.sqrt(a)
    a = a - a.mean(axis=0)
    cov = a.T @ a
    #return cov * (1/d)
    return np.eye(d) * s


if __name__ == '__main__':
    main()
