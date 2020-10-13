import numpy as np


def moments_wiener(t: np.ndarray, init_x: float = 0, mu: float = 0) -> (np.ndarray, np.ndarray):
    """
    :param t:
    :param init_x:
    :param mu:
    :return:

    >>> t = np.array([0.25, 0.75])
    >>> init_x = 1
    >>> mu = -1
    >>> moments_wiener(t, init_x, mu)
    (array([0.75, 0.25]), array([[0.25, 0.25],
           [0.25, 0.75]]))
    """

    assert 0 < np.min(t)

    tmat = np.meshgrid(t, t)
    mean = init_x + mu * t
    cov = np.min(tmat, 0)

    return mean, cov


def moments_brownbr(t: np.ndarray, fin_t: float = 1, init_x: float = 0, fin_x: float = 0) -> (np.ndarray, np.ndarray):
    """
    :param t:
    :param fin_t:
    :param init_x:
    :param fin_x:
    :return:

    >>> t = np.array([0.25, 0.75])
    >>> moments_brownbr(t)
    (array([0., 0.]), array([[0.1875, 0.0625],
           [0.0625, 0.1875]]))
    """

    assert 0 < np.min(t) <= np.max(t) < fin_t

    tmat = np.meshgrid(t, t)
    mean = init_x + (fin_x - init_x) * t / fin_t
    cov = (fin_t - np.max(tmat, 0)) * np.min(tmat, 0) / fin_t

    return mean, cov


def moments_gaussian_bridge(fin_x: float, umean: np.ndarray, ucov: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    :param fin_x:
    :param umean:
    :param ucov:
    :return:
    """

    cmean = umean[:-1] + ucov[-1, :-1] * (fin_x - umean[-1]) / ucov[-1, -1]
    ccov = ucov[:-1, :-1] - np.outer(ucov[-1, :-1], ucov[-1, :-1]) / ucov[-1, -1]

    return cmean, ccov
