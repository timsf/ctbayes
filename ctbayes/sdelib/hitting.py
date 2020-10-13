from itertools import count, dropwhile, islice
from typing import Iterator

import numpy as np

from ctbayes.misc.exceptions import BudgetConstraintError


def cdf_brownbr_min(min_x: float, fin_t: float, init_x: float, fin_x: float,
                    lb_min_x: float = None, ub_min_x: float = None) -> float:
    """
    :param min_x:
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param lb_min_x:
    :param ub_min_x:
    :return:

    >>> # generate fixture
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.normal()
    >>> fin_x = np.random.normal(init_x, np.sqrt(fin_t))
    >>> lb_min_x = min(init_x, fin_x) - np.sqrt(fin_t)
    >>> ub_min_x = min(init_x, fin_x) - .5 * np.sqrt(fin_t)

    >>> # test edge cases
    >>> np.allclose(cdf_brownbr_min(-np.inf, fin_t, init_x, fin_x), 0)
    True
    >>> np.allclose(cdf_brownbr_min(min(init_x, fin_x), fin_t, init_x, fin_x), 1)
    True
    >>> np.allclose(cdf_brownbr_min(lb_min_x, fin_t, init_x, fin_x, lb_min_x, ub_min_x), 0)
    True
    >>> np.allclose(cdf_brownbr_min(ub_min_x, fin_t, init_x, fin_x, lb_min_x, ub_min_x), 1)
    True
    """

    assert fin_t > 0
    if lb_min_x is None:
        lb_min_x = -np.inf
    if ub_min_x is None:
        ub_min_x = min(init_x, fin_x)
    assert lb_min_x < ub_min_x <= min(init_x, fin_x)

    pm = np.exp(-2 * (init_x - min_x) * (fin_x - min_x) / fin_t) if min_x < min(init_x, fin_x) else 1
    if np.isinf(lb_min_x) and ub_min_x == min(init_x, fin_x):
        return pm

    cdf_lb, cdf_ub = [cdf_brownbr_min(x, fin_t, init_x, fin_x) for x in (lb_min_x, ub_min_x)]
    p = (pm - cdf_lb) / (cdf_ub - cdf_lb)
    return p


def ppf_brownbr_min(p: float, fin_t: float, init_x: float, fin_x: float, lb_min_x: float = None, ub_min_x: float = None
                    ) -> float:
    """
    :param p:
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param lb_min_x:
    :param ub_min_x:
    :return:

    >>> # generate fixture
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.normal()
    >>> fin_x = np.random.normal(init_x, np.sqrt(fin_t))
    >>> lb_min_x = min(init_x, fin_x) - np.sqrt(fin_t)
    >>> ub_min_x = min(init_x, fin_x) - .5 * np.sqrt(fin_t)

    >>> # test edge cases
    >>> np.isinf(ppf_brownbr_min(0, fin_t, init_x, fin_x))
    True
    >>> np.allclose(ppf_brownbr_min(1, fin_t, init_x, fin_x), min(init_x, fin_x))
    True
    >>> np.allclose(ppf_brownbr_min(0, fin_t, init_x, fin_x, lb_min_x, ub_min_x), lb_min_x)
    True
    >>> np.allclose(ppf_brownbr_min(1, fin_t, init_x, fin_x, lb_min_x, ub_min_x), ub_min_x)
    True
    """

    assert 0 <= p <= 1
    if lb_min_x is None:
        lb_min_x = -np.inf
    if ub_min_x is None:
        ub_min_x = min(init_x, fin_x)
    assert 0 < fin_t
    assert lb_min_x < ub_min_x <= min(init_x, fin_x)

    if np.isinf(lb_min_x) and ub_min_x == min(init_x, fin_x):
        pm = p
    else:
        cdf_lb, cdf_ub = [cdf_brownbr_min(x, fin_t, init_x, fin_x) for x in (lb_min_x, ub_min_x)]
        pm = p * (cdf_ub - cdf_lb) + cdf_lb

    min_x = (fin_x + init_x - np.sqrt((fin_x - init_x) ** 2 - 2 * fin_t * np.log(pm))) / 2
    return min_x


def sample_brownbr_min(fin_t: float, init_x: float, fin_x: float, lb_min_x: float = None, ub_min_x: float = None,
                       ome: np.random.Generator = np.random.default_rng()) -> (float, float):
    """Sample the minimum value and its associated time from a Brownian bridge process.

    :param fin_t:
    :param init_x:
    :param fin_x:
    :param lb_min_x:
    :param ub_min_x:
    :param ome:
    :return:

    >>> # generate fixture
    >>> n = int(1e5)
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.normal()
    >>> fin_x = np.random.normal(init_x, np.sqrt(fin_t))

    >>> # draw iid samples
    >>> sample = np.array([sample_brownbr_min(fin_t, init_x, fin_x)[0] for _ in range(n)])

    >>> # test against discrete approximation
    >>> from scipy.stats import epps_singleton_2samp
    >>> from ctbayes.sdelib.paths import sample_brownbr
    >>> alpha = 1e-2
    >>> res = 1 / n
    >>> t = np.arange(res, fin_t, res)
    >>> new_x = np.array([sample_brownbr(t, fin_t, init_x, fin_x) for _ in range(n)])
    >>> ref_sample = t[new_x.argmin(1)]
    >>> alpha < epps_singleton_2samp(sample, ref_sample)[0]
    True
    """

    if lb_min_x is None:
        lb_min_x = -np.inf
    if ub_min_x is None:
        ub_min_x = min(init_x, fin_x)
    assert 0 < fin_t
    assert lb_min_x < ub_min_x <= min(init_x, fin_x)

    # simulate minimum value
    min_x = ppf_brownbr_min(ome.uniform(), fin_t, init_x, fin_x, lb_min_x, ub_min_x)

    # simulate hitting time of minimum value
    par1 = (fin_x - min_x) ** 2 / (2 * fin_t)
    par2 = (min_x - init_x) ** 2 / (2 * fin_t)
    par3 = np.sqrt(par1 / par2)
    if ome.uniform() < 1 / (1 + par3):
        min_t = fin_t / (1 + ome.wald(par3, 2 * par1))
    else:
        min_t = fin_t / (1 + 1 / ome.wald(1 / par3, 2 * par2))

    return min_t, min_x


def sample_layerbr_min(fin_t: float, init_x: float, fin_x: float,
                       ub_x: float = None, lb_min_x: float = None, ub_min_x: float = None,
                       ome: np.random.Generator = np.random.default_rng(),
                       max_props: int = int(1e6)) -> (float, float):
    """Sample the minimum value and its associated time from a layered bridge process.

    :param fin_t:
    :param init_x:
    :param fin_x:
    :param ub_x:
    :param lb_min_x:
    :param ub_min_x:
    :param ome:
    :param max_props:
    :return:
    :raise: BudgetConstraintError

    >>> # generate fixture
    >>> from ctbayes.sdelib.transforms import reflect
    >>> n = int(1e5)
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.normal()
    >>> fin_x = np.random.normal(init_x, np.sqrt(fin_t))
    >>> sup_x = init_x + fin_x - ppf_brownbr_min(0.75, fin_t, -fin_x, -init_x)

    >>> # draw iid samples
    >>> sample = np.array([sample_layerbr_min(fin_t, init_x, fin_x, sup_x)[0] for _ in range(n)])

    >>> # test against discrete approximation
    >>> from scipy.stats import epps_singleton_2samp
    >>> from ctbayes.sdelib.paths import sample_brownbr
    >>> alpha = 1e-2
    >>> res = 1 / n
    >>> t = np.arange(res, fin_t, res)
    >>> xt = np.array([sample_brownbr(t, fin_t, init_x, fin_x) for _ in range(n)])
    >>> xt = xt[np.all(xt < sup_x, 1)]
    >>> ref_sample = t[xt.argmin(1)]
    >>> alpha < epps_singleton_2samp(sample, ref_sample)[0]
    True
    """

    assert 0 < fin_t
    if ub_x is not None:
        assert max(init_x, fin_x) <= ub_x

    for _ in range(max_props):
        min_t, min_x = sample_brownbr_min(fin_t, init_x, fin_x, lb_min_x, ub_min_x, ome)
        if ub_x is None:
            return min_t, min_x
        esc1 = sample_bessel3br_esc(min_t, 0, init_x - min_x, ub_x - min_x, ome=ome)
        esc2 = sample_bessel3br_esc(fin_t - min_t, 0, fin_x - min_x, ub_x - min_x, ome=ome)
        if not esc1 and not esc2:
            return min_t, min_x
    else:
        raise BudgetConstraintError('None of the proposals were accepted.')


def sample_brownbr_esc(fin_t: float, init_x: float, fin_x: float, ue_x: float, ub_x: float = None, seed: float = None,
                       ome: np.random.Generator = np.random.default_rng()) -> bool:
    """
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param ue_x:
    :param ub_x:
    :param seed:
    :return:

    >>> # generate sample, unbounded case
    >>> from ctbayes.sdelib.paths import sample_brownbr
    >>> n = int(1e5)
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.normal()
    >>> fin_x = np.random.normal(scale=np.sqrt(fin_t))
    >>> inf_x = ppf_brownbr_min(0.75, fin_t, init_x, fin_x)
    >>> ub_sup_x = max(abs(inf_x), abs(init_x + fin_x - inf_x))
    >>> sample = [sample_brownbr_esc(fin_t, init_x, fin_x, ub_sup_x) for _ in range(n)]

    >>> # test against discrete approximation
    >>> from scipy.stats import binom_test
    >>> alpha = 1e-2
    >>> gen = series_brownbr_esc(fin_t, init_x, fin_x, ub_sup_x)
    >>> for _ in range(n): p = next(gen)
    >>> alpha < binom_test(sum(sample), len(sample), p)
    True

    >>> # generate sample, bounded case
    >>> sample = [sample_brownbr_esc(fin_t, init_x, fin_x, ub_sup_x, 1.1 * ub_sup_x) for _ in range(n)]

    >>> # test against discrete approximation
    >>> gen = series_brownbr_esc(fin_t, init_x, fin_x, ub_sup_x, 1.1 * ub_sup_x)
    >>> for _ in range(n): p = next(gen)
    >>> alpha < binom_test(sum(sample), len(sample), p)
    True
    """

    if seed is None:
        seed = ome.uniform()

    assert 0 < fin_t
    assert 0 <= seed <= 1

    if ue_x <= max(abs(init_x), abs(fin_x)):
        return True
    if ub_x is not None and ub_x < ue_x:
        return False

    for i, s in enumerate(series_brownbr_esc(fin_t, init_x, fin_x, ue_x, ub_x), 1):
        if (not i % 2 and s > seed) or (i % 2 and s < seed):
            return not bool(i % 2)


def sample_bessel3br_esc(fin_t: float, init_x: float, fin_x: float, ue_x: float, ub_x: float = None, seed: float = None,
                         ome: np.random.Generator = np.random.default_rng()) -> bool:
    """
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param ue_x:
    :param ub_x:
    :param seed:
    :return:

    >>> # generate sample, unbounded case
    >>> from ctbayes.sdelib.paths import sample_besselbr
    >>> n = int(1e5)
    >>> fin_t = np.random.uniform()
    >>> fin_x = np.sqrt(np.random.chisquare(3) * fin_t)
    >>> ub_sup_x = fin_x - ppf_brownbr_min(0.75, fin_t, -fin_x, 0)
    >>> sample = [sample_bessel3br_esc(fin_t, 0, fin_x, ub_sup_x) for _ in range(n)]

    >>> # test against discrete approximation
    >>> from scipy.stats import binom_test
    >>> alpha = 1e-2
    >>> gen = series_bessel3br_esc(fin_t, 0, fin_x, ub_sup_x)
    >>> for _ in range(n): p = next(gen)
    >>> alpha < binom_test(sum(sample), len(sample), p)
    True

    >>> # generate sample, bounded case
    >>> sample = [sample_bessel3br_esc(fin_t, 0, fin_x, ub_sup_x, 1.1 * ub_sup_x) for _ in range(n)]

    >>> # test against discrete approximation
    >>> gen = series_bessel3br_esc(fin_t, 0, fin_x, ub_sup_x, 1.1 * ub_sup_x)
    >>> for _ in range(n): p = next(gen)
    >>> alpha < binom_test(sum(sample), len(sample), p)
    True

    >>> # generate sample, unbounded non-origin case
    >>> sample = [sample_bessel3br_esc(fin_t, fin_x / 2, fin_x, ub_sup_x) for _ in range(n)]

    >>> # test against discrete approximation
    >>> gen = series_bessel3br_esc(fin_t, fin_x / 2, fin_x, ub_sup_x)
    >>> for _ in range(n): p = next(gen)
    >>> alpha < binom_test(sum(sample), len(sample), p)
    True

    >>> # generate sample, bounded fast forward case
    >>> fin_t = 3 * ub_sup_x ** 2 * 1.1
    >>> sample = [sample_bessel3br_esc(fin_t, 0, fin_x, ub_sup_x, 1.1 * ub_sup_x) for _ in range(n)]

    >>> # test against discrete approximation
    >>> gen = series_bessel3br_esc(fin_t, 0, fin_x, ub_sup_x, 1.1 * ub_sup_x)
    >>> for _ in range(n): p = next(gen)
    >>> alpha < binom_test(sum(sample), len(sample), p)
    True
    """

    if seed is None:
        seed = ome.uniform()

    assert 0 < fin_t
    assert 0 <= init_x
    assert 0 < fin_x
    assert 0 <= seed <= 1

    if ue_x <= max(init_x, fin_x):
        return True
    if ub_x is not None and ub_x < ue_x:
        return False

    gen = series_bessel3br_esc(fin_t, init_x, fin_x, ue_x, ub_x)

    # fast forward to reach cauchy sequence
    i0 = 1
    if not 3 * ue_x ** 2 > fin_t:
        seq = [0]
        for i, s in enumerate(gen, i0):
            seq.append(s)
            if (i > 2 and i % 2 and seq[i] < 1 and seq[i - 2] > seq[i] > seq[i - 1]) or (seq[i] == seq[i - 1]):
                i0 = i + 1
                break

    for i, s in enumerate(gen, i0):
        if (not i % 2 and s > seed) or (i % 2 and s < seed):
            return not bool(i % 2)


def series_brownbr_esc(fin_t: float,
                       init_x: float,
                       fin_x: float,
                       ue_x: float,
                       ub_x: float = None) -> Iterator[float]:
    """
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param ue_x:
    :param ub_x:
    :return:
    """

    def up(j: int, a: float, b: float) -> float:
        return np.exp(-2 * (ue_x * (2 * j - 1) - a) * (ue_x * (2 * j - 1) - b) / fin_t)

    def down(j: int, a: float, b: float) -> float:
        return np.exp(-2 * j * (4 * ue_x ** 2 * j + 2 * ue_x * (a - b)) / fin_t)

    assert 0 < fin_t
    assert 0 < ue_x

    if ub_x is not None:
        assert 0 < ub_x

    # handle bounded case recursively
    if ub_x is not None:
        num_seq = series_brownbr_esc(fin_t, init_x, fin_x, ue_x)
        den_seq = islice(series_brownbr_esc(fin_t, init_x, fin_x, ub_x), 1, None)
        for r, s in zip(num_seq, den_seq):
            yield 1 - (1 - r) / (1 - s)

    # handle unbounded case
    s = 0
    for i in count(1):
        if i % 2:
            s += up((i + 1) // 2, init_x, fin_x) + up((i + 1) // 2, -init_x, -fin_x)
        else:
            s -= (down(i // 2, init_x, fin_x) + down(i // 2, -init_x, -fin_x))
        yield s


def series_bessel3br_esc(fin_t: float,
                         init_x: float,
                         fin_x: float,
                         ue_x: float,
                         ub_x: float = None) -> Iterator[float]:
    """
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param ue_x:
    :param ub_x:
    :return:
    """

    def inc(i: int, b: float) -> float:
        return (2 * ue_x * i + b) * np.exp(-2 * ue_x * i * (ue_x * i + b) / fin_t)

    assert 0 < fin_t
    assert 0 <= init_x
    assert 0 < fin_x
    assert 0 < ue_x
    if ub_x is not None:
        assert 0 < ub_x

    # handle trivial case
    if min(init_x, fin_x) <= ue_x <= max(init_x, fin_x):
        return True

    # handle unbounded case recursively
    if ub_x is not None:
        num_seq = series_bessel3br_esc(fin_t, init_x, fin_x, ue_x)
        den_seq = islice(series_bessel3br_esc(fin_t, init_x, fin_x, ub_x), 1, None)
        for r, s in zip(num_seq, den_seq):
            yield 1 - (1 - r) / (1 - s)

    # handle unbounded case starting from positive value
    if init_x != 0:
        for s in series_brownbr_esc(fin_t, init_x - ue_x / 2, fin_x - ue_x / 2, ue_x / 2):
            yield 1 - (1 - s) / (1 - np.exp(-2 * init_x * fin_x / fin_t))

    # handle unbounded case starting from 0
    s = 0
    for j in count(1):
        if j % 2:
            s += inc((j + 1) // 2, -fin_x) / fin_x
        else:
            s -= inc(j // 2, fin_x) / fin_x
        yield s
