from typing import List, Optional
from itertools import chain

import numpy as np

from ctbayes.sdelib import paths


def fill_wiener_vector(t: np.ndarray, x: np.ndarray, new_t: np.ndarray,
                       ome: np.random.Generator = np.random.default_rng()) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    :param t:
    :param x:
    :param new_t:
    :param ome:
    :return:

    >>> # generate fixture
    >>> n = int(1e4)
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.normal()
    >>> fin_x = np.random.normal(init_x, np.sqrt(fin_t))
    >>> base_t = np.sort(np.random.uniform(high=fin_t, size=(n, 10)), 1)
    >>> bridges = [fill_wiener_vector(np.array([0, fin_t]), np.array([(init_x,), (fin_x,)]), t_) for t_ in base_t]
    >>> t, yt, _ = zip(*bridges)
    >>> test_t = np.sort(np.random.uniform(high=fin_t, size=10))

    >>> # draw iid samples
    >>> bridges = [fill_wiener_vector(t_, x_, test_t) for t_, x_ in zip(t, yt)]
    >>> test_xt = np.hstack([x_[i_] for _, x_, i_ in bridges])

    >>> from scipy.stats import kstest, chi2
    >>> from ctbayes.sdelib.moments import moments_brownbr
    >>> from ctbayes.misc.linalg import eval_quad
    >>> alpha = 1e-2
    >>> true_mean, true_cov = moments_brownbr(test_t, fin_t, init_x, fin_x)
    >>> sample = eval_quad(test_xt.T - true_mean, true_cov)
    >>> dist = chi2(len(test_t))
    >>> alpha < kstest(sample, dist.cdf)[1]
    True
    """

    assert 0 <= np.min(t) < np.min(new_t) <= np.max(new_t) <= np.max(t)
    assert t.shape[0] == x.shape[0]

    if len(new_t) == 0:
        return t, x, np.int64([])

    insert_to = np.searchsorted(t, new_t, side='right')
    sectors = np.unique(insert_to)

    new_x = np.vstack([paths.sample_brownbr(new_t[insert_to == i] - t[i - 1], t[i] - t[i - 1], x[i - 1], x[i], ome)
                       for i in sectors])

    stack_t = np.insert(np.float64(t), insert_to, new_t)
    stack_x = np.insert(np.float64(x), insert_to, new_x, 0)

    return stack_t, stack_x, insert_to + np.arange(len(insert_to))


def fill_wiener_outer(t: np.ndarray, x: np.ndarray, new_t: np.ndarray, min_x: float,
                      ue_x: float = None, hit_x: float = None, hit_i: List[bool] = None,
                      ome: np.random.Generator = np.random.default_rng()
                      ) -> (np.ndarray, np.ndarray, np.ndarray, Optional[List[bool]]):
    """
    :param t:
    :param x:
    :param new_t:
    :param min_x:
    :param ue_x:
    :param hit_x:
    :param hit_i:
    :param ome:
    :return:

    >>> # generate fixture
    >>> from ctbayes.sdelib.hitting import ppf_brownbr_min
    >>> n = int(1e3)
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.normal()
    >>> fin_x = init_x
    >>> inf_x = ppf_brownbr_min(0.75, fin_t, init_x, fin_x)
    >>> ub_sup_x = init_x + fin_x - inf_x
    >>> s = np.sort(np.random.uniform(high=fin_t, size=(n, 2)), 1)
    >>> xt = np.array([paths.sample_layerbr(s_, fin_t, init_x, fin_x, inf_x, ub_sup_x)[0] for s_ in s])
    >>> t = np.random.uniform(high=fin_t, size=1)

    >>> # draw iid samples
    >>> bridges = [fill_wiener_outer(np.hstack([0, s_, fin_t]), np.hstack([init_x, xt_, fin_x]), t, inf_x, ub_sup_x) for s_, xt_ in zip(s, xt)]
    >>> xt = np.array([xt_[i_] for t_, xt_, i_, _ in bridges])

    >>> # test against direct sampling
    >>> from scipy.stats import epps_singleton_2samp
    >>> alpha = 1e-2
    >>> yt = np.array([paths.sample_layerbr(t, fin_t, init_x, fin_x, inf_x, ub_sup_x)[0] for _ in range(n)])
    >>> ref_sample = yt.flatten()
    >>> sample = xt.flatten()
    >>> alpha < epps_singleton_2samp(sample, ref_sample)[1]
    True
    """

    assert 0 <= np.min(t) <= np.min(new_t) <= np.max(new_t) <= np.max(t)
    assert min_x <= np.min(x)
    if ue_x is not None:
        assert np.max(x) < ue_x
        if hit_x is not None:
            assert min_x < hit_x < ue_x

    present_t = np.isin(new_t, t)
    stack_t, stack_x, new_sectors, stack_hit_i = fill_wiener_inner(t, x, new_t[~present_t], min_x, ue_x, hit_x, hit_i, ome)

    return stack_t, stack_x, np.where(np.isin(stack_t, new_t))[0], stack_hit_i


def fill_wiener_inner(t: np.ndarray, x: np.ndarray, new_t: np.ndarray, min_x: float,
                      ue_x: float = None, hit_x: float = None, hit_i: List[bool] = None,
                      ome: np.random.Generator = np.random.default_rng()) -> (np.ndarray, np.ndarray, np.ndarray, Optional[List[bool]]):
    """
    :param t:
    :param x:
    :param new_t:
    :param min_x:
    :param ue_x:
    :param hit_x:
    :param hit_i:
    :param ome:
    :return:
    """

    if not len(new_t):
        return t, x, np.int64([]), None

    assert 0 <= np.min(t) < np.min(new_t) <= np.max(new_t) < np.max(t)
    assert min_x <= np.min(x)
    if ue_x is not None:
        assert np.max(x) < ue_x
        if hit_x is not None:
            assert min_x < hit_x < ue_x

    insert_to = np.searchsorted(t, new_t, side='left')
    sectors = np.unique(insert_to)

    if hit_i is None:
        ue_x_sec = (len(t) - 1) * [ue_x]
        hit_x_sec = (len(t) - 1) * [None]
    else:
        ue_x_sec = [ue_x if hit_sec else hit_x for hit_sec in hit_i]
        hit_x_sec = [hit_x if hit_sec else None for hit_sec in hit_i]

    new_x, new_hits = zip(*[paths.sample_layerbr(
        new_t[insert_to == i] - t[i-1], t[i] - t[i-1], x[i-1], x[i], min_x, ue_x_sec[i-1], hit_x_sec[i-1], ome)
        for i in sectors])

    stack_t = np.insert(np.float64(t), insert_to, new_t)
    stack_x = np.insert(np.float64(x), insert_to, np.hstack(new_x))

    if hit_i is None:
        stack_hit_i = None
    else:
        stack_hit_i = [(h,) for h in hit_i]
        for i, a, b in zip(sectors - 1, new_hits, new_x):
            if a is None:
                stack_hit_i[i] *= (len(b) + 1)
            else:
                stack_hit_i[i] = tuple(a)
        stack_hit_i = list(chain.from_iterable(stack_hit_i))

    return stack_t, stack_x, insert_to + np.arange(len(insert_to)), stack_hit_i
