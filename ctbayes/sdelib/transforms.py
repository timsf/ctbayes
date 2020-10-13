from typing import Callable, Optional
from ctbayes.sdelib.hitting import ppf_brownbr_min

import numpy as np


def reverse(t: np.ndarray, x: np.ndarray, fin_t: float = 1) -> (np.ndarray, np.ndarray):
    """
    :param t:
    :param x:
    :param fin_t:
    :return:

    >>> fin_t = np.random.uniform()
    >>> t = np.sort(np.random.uniform(high=fin_t, size=10))
    >>> y = np.random.standard_normal(10)
    >>> t_, x_ = reverse(*reverse(t, y, fin_t), fin_t)
    >>> np.allclose(t, t_), np.allclose(y, x_)
    (True, True)
    """

    if len(t):
        if not 0 <= np.min(t) <= np.max(t) <= fin_t:
            raise Exception
        assert 0 <= np.min(t) <= np.max(t) <= fin_t
    assert 0 <= fin_t

    return fin_t - t[::-1], x[::-1]


def reflect(t: Optional[np.ndarray], x: np.ndarray, reflect_x: float = 0) -> (np.ndarray, np.ndarray):
    """
    :param t:
    :param x:
    :param reflect_x:
    :return:

    >>> reflect_x = np.random.normal()
    >>> t = np.sort(np.random.uniform(high=np.random.uniform(), size=10))
    >>> y = np.random.standard_normal(10)
    >>> t_, x_ = reflect(*reflect(t, y, reflect_x), reflect_x)
    >>> np.allclose(t, t_), np.allclose(y, x_)
    (True, True)
    """

    if t is not None:
        assert 0 <= np.min(t)

    return t, 2 * reflect_x - x


def pivot_brownbr(s: np.ndarray,
                  z: np.ndarray,
                  fin_t: float,
                  init_x: float,
                  fin_x: float,
                  norm_time: Callable[[float], float] = lambda x: x,
                  denorm_time: Callable[[float], float] = lambda x: x) -> (np.ndarray, np.ndarray):
    """
    :param s:
    :param z:
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param norm_time:
    :param denorm_time
    :return:
    """

    assert 0 <= np.min(s) <= np.max(s) <= 1
    assert 0 < fin_t

    amp = norm_time(fin_t)
    t = np.array([denorm_time(s_ * amp) for s_ in s])
    x = z * np.sqrt(amp) + init_x + s * (fin_x - init_x)

    return t, x


def gen_seeded_bessel3br(s: np.ndarray,
                         z: np.ndarray,
                         fin_t: float,
                         fin_x: float) -> (np.ndarray, np.ndarray):
    """
    :param s:
    :param z:
    :param fin_t:
    :param fin_x:
    :return:

    >>> # generate fixture
    >>> from ctbayes.sdelib.paths import sample_brownbr, sample_besselbr
    >>> n = int(1e4)
    >>> fin_t = np.random.uniform()
    >>> fin_x = np.sqrt(np.random.chisquare(3) * fin_t)
    >>> test_s = np.sort(np.random.uniform(size=1))

    >>> # draw iid sample
    >>> new_x = [sample_brownbr(test_s, 1, np.zeros(3), np.zeros(3)) for _ in range(n)]
    >>> test_t, y = zip(*[gen_seeded_bessel3br(test_s, z_, fin_t, fin_x) for z_ in new_x])
    >>> test_t, y = np.array(test_t[:1]), np.array(y)

    >>> # test against reference implementation
    >>> from scipy.stats import epps_singleton_2samp
    >>> alpha = 1e-2
    >>> y = sample_besselbr(test_t, fin_t, np.repeat(fin_x, n), 3)
    >>> sample = y.flatten()
    >>> ref_sample = y.flatten()
    >>> (alpha < epps_singleton_2samp(sample, ref_sample)[1], alpha < epps_singleton_2samp(sample, ref_sample)[1])
    (True, True)
    """

    assert 0 < np.min(s) <= np.max(s) < 1
    assert 0 < fin_t

    t = s * fin_t
    z = np.hstack([(z[:, 0] + fin_x * t / fin_t ** 1.5)[:, np.newaxis], z[:, 1:]])

    return t, np.sqrt(fin_t) * np.linalg.norm(z, axis=1)


def gen_seeded_brownbr_min(u: (float, float, float),
                           r2: float,
                           fin_t: float = 1,
                           init_x: float = 0,
                           fin_x: float = 0) -> (float, float):
    """For a given Wiener ome_nil_, compute its minimum value for any endpoint transformation.

    :param u:
    :param r2:
    :param fin_t:
    :param init_x:
    :param fin_x:
    :return:

    >>> # generate fixture
    >>> n = int(1e4)
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.lognormal()
    >>> fin_x = np.random.lognormal()
    >>> u = np.random.uniform(size=(n, 3))
    >>> r2 = np.random.chisquare(1, size=n)

    >>> # draw iid sample
    >>> sample = np.array([gen_seeded_brownbr_min(u_, r2_, fin_t, init_x, fin_x) for u_, r2_ in zip(u, r2)]).T

    >>> # test against reference implementation
    >>> from ctbayes.sdelib.hitting import sample_brownbr_min
    >>> from scipy.stats import epps_singleton_2samp
    >>> alpha = 1e-2
    >>> ref_sample = np.array([sample_brownbr_min(fin_t, init_x, fin_x) for i in range(n)]).T
    >>> (alpha < epps_singleton_2samp(sample[0], ref_sample[0])[1], alpha < epps_singleton_2samp(sample[1], ref_sample[1])[1])
    (True, True)
    """

    def gen_seeded_wald(mu: float, lam: float) -> float:
        cand = mu + (mu ** 2 * r2 - mu * np.sqrt(4 * mu * lam * r2 + mu ** 2 * r2 ** 2)) / (2 * lam)
        if u[1] < mu / (mu + cand):
            return cand
        return mu ** 2 / cand

    assert 0 < np.min(u) <= np.max(u) < 1
    assert 0 < r2
    assert 0 < fin_t

    min_x = ppf_brownbr_min(u[0], fin_t, init_x, fin_x)

    par1 = (fin_x - min_x) ** 2 / (2 * fin_t)
    par2 = (min_x - init_x) ** 2 / (2 * fin_t)
    par3 = np.sqrt(par1 / par2)
    if u[2] < 1 / (1 + par3):
        min_t = fin_t / (1 + gen_seeded_wald(par3, 2 * par1))
    else:
        min_t = fin_t / (1 + 1 / gen_seeded_wald(1 / par3, 2 * par2))

    return min_t, min_x
