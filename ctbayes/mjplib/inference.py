from typing import List

import numpy as np

from ctbayes.mjplib.skeleton import Skeleton


def update(skel: Skeleton, alp0: np.ndarray, bet0: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    :param skel:
    :param alp0:
    :param bet0:
    :return:

    >>> # generate fixture
    >>> from ctbayes.mjplib.skeleton import sample_forwards
    >>> dim = 3
    >>> fin_t = 1e5
    >>> alp0 = np.ones((dim, dim))
    >>> np.fill_diagonal(alp0, np.repeat(np.nan, dim))
    >>> bet0 = np.ones((dim, dim))
    >>> np.fill_diagonal(bet0, np.repeat(np.nan, dim))
    >>> gen = get_ev(alp0, bet0)
    >>> skel = sample_forwards(fin_t, None, gen)

    >>> # test posterior concentration
    >>> alp1, bet1 = update(skel, alp0, bet0)
    >>> est_gen = get_ev(alp1, bet1)
    >>> np.allclose(gen, est_gen, 1e-1)
    True
    """

    dim = bet0.shape[0]
    trans = np.float64([np.bincount(skel.xt[np.hstack([False, (skel.xt == i)[:-1]])], minlength=dim) for i in range(dim)])
    np.fill_diagonal(trans, np.nan)
    hold = np.array([np.sum(np.ediff1d(skel.t, to_end=(0,))[skel.xt == i]) for i in range(dim)])
    return trans + alp0, (hold + bet0.T).T


def sample_param(alp: np.ndarray, bet: np.ndarray, ome: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    """
    :param alp:
    :param bet:
    :param ome:
    :return:
    """

    gen = ome.gamma(alp, 1 / bet)
    np.fill_diagonal(gen, -np.nansum(gen, 1))
    return gen


def eval_loglik(skel: Skeleton, gen: np.ndarray) -> float:
    """
    :param skel:
    :param gen:
    :return:
    """

    return np.sum(np.log(gen[skel.xt[:-1], skel.xt[1:]]) + (skel.t[1:] - skel.t[:-1]) * np.diag(gen)[skel.xt[:-1]]).item()


def eval_logprior(gen: np.ndarray, alp: np.ndarray, bet: np.ndarray) -> float:
    """
    :param gen:
    :param alp:
    :param bet:
    :return:
    """

    masked_gen = np.where(gen > 0, gen, np.nan)
    return np.nansum((alp - 1) * np.log(masked_gen) - bet * masked_gen)


def get_ev(alp: np.ndarray, bet: np.ndarray) -> np.ndarray:
    """
    :param alp:
    :param bet:
    :return:
    """

    gen = alp / bet
    np.fill_diagonal(gen, -np.nansum(gen, 1))
    return gen


def get_mode(alp: np.ndarray, bet: np.ndarray) -> np.ndarray:
    """
    :param alp:
    :param bet:
    :return:
    """

    gen = (alp - 1) / bet
    np.fill_diagonal(gen, -np.nansum(gen, 1))
    return gen
