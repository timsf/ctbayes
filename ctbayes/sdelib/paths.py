from typing import List, Optional

import numpy as np

from ctbayes.sdelib import hitting, moments
from ctbayes.misc.exceptions import BudgetConstraintError


def sample_wiener(t: np.ndarray, init_x: np.ndarray, mu: np.ndarray = None,
                  ome: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    """Sample from a multivariate Wiener process at given times.

    :param t:
    :param init_x:
    :param mu:
    :param ome:
    :return:

    >>> # generate fixture
    >>> n = int(1e5)
    >>> init_x = np.random.normal(n)
    >>> mu = np.random.normal(n)
    >>> t = np.sort(np.random.uniform(size=10))

    >>> # draw iid samples
    >>> xt = sample_wiener(t, np.repeat(init_x, n), np.repeat(mu, n))

    >>> # test mahalanobis distance sampling distribution
    >>> from scipy.stats import kstest, norm
    >>> alpha = 1e-2
    >>> true_mean, true_cov = moments.moments_wiener(t, init_x, mu)
    >>> sample = np.linalg.solve(np.linalg.cholesky(true_cov), xt - true_mean[:, np.newaxis]).flatten()
    >>> alpha < kstest(sample, norm().cdf)[1]
    True
    """

    assert 0 < np.min(t)

    dt = np.ediff1d(t, to_begin=(t[0],))
    dxt = np.sqrt(dt)[:, np.newaxis] * ome.standard_normal((len(t), len(init_x)))
    if mu is not None:
        dxt += np.outer(dt, mu)
    xt = np.cumsum(dxt, axis=0) + init_x

    return xt


def sample_brownbr(t: np.ndarray, fin_t: float, init_x: float, fin_x: float,
                   ome: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    """Sample from a Brownian bridge process at given times.

    :param t:
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param ome:
    :return:

    >>> # generate fixture
    >>> n = int(1e4)
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.normal()
    >>> fin_x = np.random.normal(init_x, np.sqrt(fin_t))
    >>> t = np.sort(np.random.uniform(high=fin_t, size=10))

    >>> # draw iid samples
    >>> xt = np.array([sample_brownbr(t, fin_t, init_x, fin_x) for _ in range(n)])

    >>> # test mahalanobis distance sampling distribution
    >>> from scipy.stats import kstest, norm
    >>> alpha = 1e-2
    >>> true_mean, true_cov = moments.moments_brownbr(t, fin_t, init_x, fin_x)
    >>> sample = np.linalg.solve(np.linalg.cholesky(true_cov), (xt - true_mean).T).flatten()
    >>> alpha < kstest(sample, norm().cdf)[1]
    True
    """

    assert 0 < np.min(t) <= np.max(t) < fin_t

    std_t = t / fin_t
    std_init_x = init_x / np.sqrt(fin_t)
    std_fin_x = fin_x / np.sqrt(fin_t)
    std_xt, = (1 - std_t) * sample_wiener(std_t / (1 - std_t), np.array([std_init_x]), np.array([std_fin_x]), ome).T
    xt = np.sqrt(fin_t) * std_xt

    return xt


def sample_bessel(t: np.ndarray, init_x: float, dim: int, mu: float = None,
                  ome: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    """Sample from a Bessel process at given times.

    :param t:
    :param init_x:
    :param dim:
    :param mu:
    :param ome:
    :return:

    >>> # generate fixture
    >>> n = int(1e4)
    >>> dim = np.random.poisson() + 1
    >>> init_x = np.random.lognormal()
    >>> t = np.random.uniform(size=1)

    >>> # draw iid samples
    >>> xt = np.array([sample_bessel(t, init_x, dim) for _ in range(n)])

    >>> # test marginal sampling distribution
    >>> from scipy.stats import kstest, ncx2
    >>> alpha = 1e-2
    >>> sample = (np.square(xt) / t).flatten()
    >>> dist = ncx2(dim, init_x ** 2 / t)
    >>> alpha < kstest(sample, dist.cdf)[1]
    True
    """

    assert 0 < np.min(t)
    assert 0 <= np.min(init_x)
    assert 0 < dim

    if mu is None:
        mu = 0
    wt = sample_wiener(t, np.append(np.zeros(dim - 1), init_x), np.append(np.zeros(dim - 1), mu), ome)

    return np.linalg.norm(wt, axis=1)


def sample_besselbr(t: np.ndarray, fin_t: float, fin_x: float, dim: int,
                    ome: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    """Sample from a Bessel bridge process at given times.

    :param t:
    :param fin_t:
    :param fin_x:
    :param dim:
    :param ome:
    :return:

    >>> # generate fixture
    >>> n = int(1e4)
    >>> dim = np.random.poisson() + 1
    >>> fin_t = np.random.uniform()
    >>> fin_x = np.sqrt(np.random.chisquare(dim, n) * fin_t)
    >>> t = np.random.uniform(high=fin_t, size=1)

    >>> # draw iid samples from marginal distribution
    >>> xt = np.array([sample_besselbr(t, fin_t, fin_x_, dim) for fin_x_ in fin_x])

    >>> # test marginal sampling distribution
    >>> from scipy.stats import kstest, chi2
    >>> alpha = 1e-2
    >>> sample = (np.square(xt) / t).flatten()
    >>> dist = chi2(dim)
    >>> alpha < kstest(sample, dist.cdf)[1]
    True
    """

    assert 0 < np.min(t) <= np.max(t) < fin_t
    assert 0 < dim

    fin_w = np.append(np.zeros(dim - 1), fin_x)
    xt = np.linalg.norm([sample_brownbr(t, fin_t, 0, fin_w_, ome) for fin_w_ in fin_w], axis=0)

    return xt


def sample_layerbr(t: np.ndarray, fin_t: float, init_x: float, fin_x: float, inf_x: float,
                   ub_sup_x: float = None, lb_sup_x: float = None, ome: np.random.Generator = np.random.default_rng(),
                   max_props: int = int(1e6)) -> (np.ndarray, Optional[List[bool]]):
    """
    :param t:
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param inf_x:
    :param ub_sup_x:
    :param lb_sup_x:
    :param max_props:
    :param ome:
    :return:
    :raise: BudgetConstraintError

    >>> # generate fixture
    >>> n = int(1e4)
    >>> res = 1 / n
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.normal()
    >>> fin_x = init_x
    >>> inf_x = hitting.ppf_brownbr_min(0.75, fin_t, init_x, fin_x)
    >>> lb_sup_x = init_x + fin_x - inf_x
    >>> ub_sup_x = 2 * lb_sup_x - max(init_x, fin_x)
    >>> t = np.arange(res, fin_t, res)
    >>> s = np.sort(np.random.choice(t, 1, False))

    >>> # draw iid samples
    >>> xt = np.array([sample_layerbr(s, fin_t, init_x, fin_x, inf_x, ub_sup_x, lb_sup_x)[0] for _ in range(n)])

    >>> # test against discrete approximation
    >>> from scipy.stats import epps_singleton_2samp
    >>> alpha = 1e-2
    >>> yt = np.array([sample_brownbr(t, fin_t, init_x, fin_x) for _ in range(n)])
    >>> yt = yt[np.all(inf_x <= yt, 1) & np.all(yt < ub_sup_x, 1) & np.any(yt > lb_sup_x, 1), np.isin(t, s)]
    >>> ref_sample = yt.flatten()
    >>> sample = xt.flatten()
    >>> alpha < epps_singleton_2samp(sample, ref_sample)[1]
    True
    """

    assert 0 < np.min(t) <= np.max(t) < fin_t
    assert 0 < max_props
    assert inf_x <= min(init_x, fin_x)
    if ub_sup_x is not None:
        assert max(init_x, fin_x) < ub_sup_x
        if lb_sup_x is not None:
            assert lb_sup_x < ub_sup_x

    for _ in range(max_props):
        xt, hit_i = propose_layerbr(t, fin_t, init_x, fin_x, inf_x, ub_sup_x, lb_sup_x, ome)
        if lb_sup_x is None:
            return xt, None
        if any(hit_i):
            return xt, hit_i
    else:
        raise BudgetConstraintError('None of the proposals were accepted.')


def propose_layerbr_edge(t: np.ndarray, fin_t: float, fin_x: float, ub_sup_x: float = None, lb_sup_x: float = None,
                         ome: np.random.Generator = np.random.default_rng(), max_props: int = int(1e6)
                         ) -> (np.ndarray, List[bool]):
    """
    :param t:
    :param fin_t:
    :param fin_x:
    :param ub_sup_x:
    :param lb_sup_x:
    :param max_props:
    :param ome:
    :return:
    :raise: BudgetConstraintError

    >>> # generate fixture
    >>> n = int(1e4)
    >>> res = 1 / n
    >>> fin_t = np.random.uniform()
    >>> fin_x = np.sqrt(np.random.chisquare(3) * fin_t)
    >>> ub_sup_x = fin_x - hitting.ppf_brownbr_min(0.75, fin_t, -fin_x, 0)
    >>> t = np.arange(res, fin_t, res)
    >>> s = np.sort(np.random.choice(t, 1, False))

    >>> # draw iid samples
    >>> xt = np.array([propose_layerbr_edge(s, fin_t, fin_x, ub_sup_x)[0] for _ in range(n)])

    >>> # test against discrete approximation
    >>> from scipy.stats import epps_singleton_2samp
    >>> alpha = 1e-2
    >>> yt = np.array([sample_besselbr(t, fin_t, fin_x, 3) for _ in range(n)])
    >>> yt = yt[np.all(yt < ub_sup_x, 1), np.isin(t, s)]
    >>> ref_sample = yt.flatten()
    >>> sample = xt.flatten()
    >>> alpha < epps_singleton_2samp(sample, ref_sample)[1]
    True
    """

    if len(t) != 0:
        assert 0 < np.min(t) <= np.max(t) < fin_t
    if ub_sup_x is not None:
        assert fin_x < ub_sup_x

    for _ in range(max_props):
        if len(t) == 0:
            xt = np.array([])
        else:
            xt = sample_besselbr(t, fin_t, fin_x, 3, ome)
        t_, xt_ = np.hstack([0, t, fin_t]), np.hstack([0, xt, fin_x])
        if ub_sup_x is None or not any([hitting.sample_bessel3br_esc(dt, x0, x1, ub_sup_x, ome=ome)
                                        for dt, x0, x1 in zip(np.diff(t_), xt_[:-1], xt_[1:])]):
            if lb_sup_x is None:
                return xt, []
            else:
                return xt, [hitting.sample_bessel3br_esc(dt, x0, x1, lb_sup_x, ub_sup_x, ome=ome)
                            for dt, x0, x1 in zip(np.diff(t_), xt_[:-1], xt_[1:])]
    else:
        raise BudgetConstraintError('None of the proposals were accepted.')


def propose_layerbr(t: np.ndarray, fin_t: float, init_x: float, fin_x: float,
                    inf_x: float, ub_sup_x: float = None, lb_sup_x: float = None,
                    ome: np.random.Generator = np.random.default_rng()) -> (np.ndarray, List[bool]):
    """
    :param t:
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param inf_x:
    :param ub_sup_x:
    :param lb_sup_x:
    :param ome:
    :return:

    >>> # generate fixture
    >>> n = int(1e4)
    >>> res = 1 / n
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.normal()
    >>> fin_x = np.random.normal(init_x, np.sqrt(fin_t))
    >>> inf_x = hitting.ppf_brownbr_min(0.75, fin_t, init_x, fin_x)
    >>> ub_sup_x = init_x + fin_x - inf_x
    >>> t = np.arange(res, fin_t, res)
    >>> s = np.sort(np.random.choice(t, 1, False))

    >>> # draw iid samples
    >>> xt = np.array([propose_layerbr(s, fin_t, init_x, fin_x, inf_x, ub_sup_x)[0] for _ in range(n)])

    >>> # test against discrete approximation
    >>> from scipy.stats import epps_singleton_2samp
    >>> alpha = 1e-2
    >>> yt = np.array([sample_brownbr(t, fin_t, init_x, fin_x) for _ in range(n)])
    >>> yt = yt[np.all((inf_x <= yt) & (yt < ub_sup_x), 1), np.isin(t, s)]
    >>> ref_sample = yt.flatten()
    >>> sample = xt.flatten()
    >>> alpha < epps_singleton_2samp(sample, ref_sample)[1]
    True
    """

    if len(t):
        assert 0 < np.min(t) <= np.max(t) < fin_t
    assert inf_x <= min(init_x, fin_x)
    if ub_sup_x is not None:
        assert max(init_x, fin_x) < ub_sup_x
        if lb_sup_x is not None:
            if not lb_sup_x < ub_sup_x:
                raise Exception
            assert lb_sup_x < ub_sup_x

    # handle edge minimum case
    if inf_x in (init_x, fin_x):
        lb_sup_x_, ub_sup_x_ = [x - inf_x if x is not None else None for x in (lb_sup_x, ub_sup_x)]
        if inf_x == fin_x:
            xt, hit_i = [out[::-1] for out in propose_layerbr_edge(fin_t - t[::-1], fin_t, init_x - inf_x,
                                                                   ub_sup_x_, lb_sup_x_, ome)]
        else:
            xt, hit_i = propose_layerbr_edge(t, fin_t, fin_x - inf_x, ub_sup_x_, lb_sup_x_, ome)
        return xt + inf_x, hit_i

    # handle exterior minimum case
    else:
        min_t, min_x = hitting.sample_layerbr_min(fin_t, init_x, fin_x, ub_sup_x, inf_x, ome=ome)
        t0, t1 = np.split(t, np.searchsorted(t, [min_t]))
        x0, i0 = propose_layerbr(t0, min_t, init_x, min_x, min_x, ub_sup_x, lb_sup_x, ome)
        x1, i1 = propose_layerbr(t1 - min_t, fin_t - min_t, min_x, fin_x, min_x, ub_sup_x, lb_sup_x, ome)
        xt = np.concatenate([x0, x1])
        if lb_sup_x is None:
            return xt, []
        return xt, i0[:-1] + [i0[-1] or i1[0]] + i1[1:]
