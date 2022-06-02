'''
Python re-implementation (from Matlab code) from the paper http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=7371221&tag=1
Fundamental matrix estimation from homographies
'''

import numpy as np


def low_rank_approx(SVD=None, A=None, r=2):
    """
    From: https://gist.github.com/thearn/5424219
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    if not SVD:
        SVD = np.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar


def normalize_linear_transform(M):
    """

    """

    m = 3
    Q = np.zeros((4, 4))
    c1 = np.sum(M[:, 0]) / m
    c2 = np.sum(M[:, 1]) / m
    c3 = np.sum(M[:, 2]) / m

    m1 = M[:, 0] - c1
    m2 = M[:, 1] - c2
    m3 = M[:, 2] - c3

    d = 0
    for i in range(0, m):
        d = d + np.sqrt(m1[i] * m1[i] + m2[i] * m2[i] + m3[i] * m3[i]);
    d = d / m
    S = np.sqrt(3) / d
    Q[0:3, 0:3] = np.eye(3) * S
    Q[3, :] = [-c1 * S, -c2 * S, -c3 * S, 1]
    return Q


def solve_ls(A):
    """
    Ax = 0 solution
    """
    U, D, V = np.linalg.svd(A)
    V = V.T
    return V[:, 3]


def calculate_F_from_3_homographies(Hs):
    """
    """

    A1, A2, A3 = [], [], []
    for i in range(0, len(Hs)):
        A1.append(Hs[i][0])
        A2.append(Hs[i][1])
        A3.append(Hs[i][2])

    A1, A2, A3 = np.array(A1).reshape(3, 3), np.array(A2).reshape(3, 3), np.array(A3).reshape(3, 3)

    A1, A2, A3 = low_rank_approx(A=A1), low_rank_approx(A=A2), low_rank_approx(A=A3)

    A1, A2, A3 = np.c_[A1, np.ones(3)], np.c_[A2, np.ones(3)], np.c_[A3, np.ones(3)]

    Q1, Q2, Q3 = normalize_linear_transform(A1), normalize_linear_transform(A2), normalize_linear_transform(A3)

    A1, A2, A3 = A1.dot(Q1), A2.dot(Q2), A3.dot(Q3)

    rank_A3 = np.linalg.matrix_rank(A3)

    if rank_A3 == 4:
        U, D, V = np.linalg.svd(A3)
        S = np.diag(D)
        S[:, 2] = 0
        A3 = np.dot(U, np.dot(S, V))

    f1, f2, f3 = solve_ls(A1), solve_ls(A2), solve_ls(A3)
    f1, f2, f3 = np.dot(Q1, f1)[:3], np.dot(Q2, f2)[:3], np.dot(Q3, f3)[:3]

    a_lambda = np.zeros((9, 3))

    for i in range(0, len(Hs)):
        h1 = Hs[i][0]
        h2 = Hs[i][1]
        h3 = Hs[i][2]

        a_lambda[3 * i - 3, :] = [f1.dot(h2), f2.dot(h1), 0]
        a_lambda[3 * i - 2, :] = [f1.dot(h3), 0, f3.dot(h1)]
        a_lambda[3 * i - 1, :] = [0, f2.dot(h3), f3.dot(h2)]

    U, D, V = np.linalg.svd(a_lambda)
    S = np.zeros((3, 9))

    S[0, 0] = D[0]
    S[1, 1] = D[1]
    S[2, 2] = 0

    a_lambda_ = np.dot(U, np.dot(S.T, V))
    U, D, V = np.linalg.svd(a_lambda_)
    lambda_ = V.T[:, 2]

    f1, f2, f3 = np.array(lambda_[0] * f1), np.array(lambda_[1] * f2), np.array(lambda_[2] * f3)

    fundamental = np.c_[f1, f2, f3]
    fundamental = fundamental / (np.linalg.norm(fundamental, 2))
    U, D, V = np.linalg.svd(fundamental)
    S = np.diag(D)
    S[:, 2] = 0

    fundamental = np.dot(U, np.dot(S, V))

    return fundamental


if __name__ == '__main__':
    '''
    3 H matrices in the format of each [[h11, h12, h13],[h21, h22, h23], [h31, h32, h33]]
    '''
    Hs = np.array(
        [
            [[1.00000236e+00, 2.94225902e-05, 1.19545967e-02],
             [1.86361252e-06, 1.00007306e+00, 2.54985438e-03],
             [-1.64121784e-08, 1.37516038e-07, 1.00000000e+00]],
            [[1.00013215e+00, 1.70836337e-04, 2.76444007e-03],
             [2.34965838e-04, 9.99993212e-01, -3.85790138e-02],
             [4.64326987e-07, -2.44171347e-07, 1.00000000e+00]],
            [[9.99989565e-01, 2.65273535e-04, -1.36674881e-02],
             [1.81349101e-04, 9.99948609e-01, -8.27417296e-03],
             [2.55651275e-07, -8.71613505e-08, 1.00000000e+00]]])

    F = calculate_F_from_3_homographies(Hs)
    print(F)