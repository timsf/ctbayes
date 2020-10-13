from itertools import count
from typing import Optional, List, Tuple

import numpy as np

from ctbayes.sdelib import hitting, paths
from ctbayes.misc.exceptions import BudgetConstraintError


def sample_layer(fin_t: float, init_x: float, fin_x: float, lb_x: float = -np.inf, ub_x: float = np.inf,
                 ome: np.random.Generator = np.random.default_rng(), max_props: int = int(1e6)) -> int:
    """
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param lb_x:
    :param ub_x:
    :param ome:
    :param max_props:
    :return:

    >>> # generate fixture, unbounded case
    >>> n = int(1e4)
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.normal()
    >>> fin_x = np.random.normal(init_x, np.sqrt(fin_t))

    >>> # draw iid samples
    >>> l = [sample_layer(fin_t, init_x, fin_x) for _ in range(n)]
    >>> sample = np.bincount(l)[1:11]
    >>> edges_x = [compute_edges(l_, fin_t, init_x, fin_x) for l_ in range(len(sample) + 1)]

    >>> # test against discrete approximation
    >>> from scipy.stats import chi2_contingency
    >>> alpha = 1e-2
    >>> res = 1 / n
    >>> t = np.arange(res, fin_t, res)
    >>> xt = np.array([paths.sample_brownbr(t, fin_t, init_x, fin_x) for _ in range(n)])
    >>> edges_x = [compute_edges(l_, fin_t, init_x, fin_x) for l_ in range(len(sample) + 1)]
    >>> ref_sample = np.ediff1d([sum(np.all((le_x < xt) & (xt < ue_x), 1)) for le_x, ue_x in edges_x])
    >>> alpha < chi2_contingency([sample, ref_sample])[1]
    True

    >>> # generate fixture, bounded cases
    >>> lb_x, ub_sup_x = compute_edges(1, 1.1 * fin_t, init_x, fin_x)

    >>> # test against discrete approximation, lower bounded case
    >>> l = [sample_layer(fin_t, init_x, fin_x, lb_x) for _ in range(n)]
    >>> sample = np.bincount(l)[1:11]
    >>> yt = xt[np.all(xt > lb_x, 1)]
    >>> edges_y = [compute_edges(l_, fin_t, init_x, fin_x, lb_x) for l_ in range(len(sample) + 1)]
    >>> ref_sample = np.ediff1d([sum(np.all((le_x < yt) & (yt < ue_x), 1)) for le_x, ue_x in edges_y])
    >>> alpha < chi2_contingency([sample, ref_sample])[1]
    True

    >>> # test against discrete approximation, upper bounded case
    >>> l = [sample_layer(fin_t, init_x, fin_x, -np.inf, ub_sup_x) for _ in range(n)]
    >>> sample = np.bincount(l)[1:11]
    >>> yt = xt[np.all(xt < ub_sup_x, 1)]
    >>> edges_y = [compute_edges(l_, fin_t, init_x, fin_x, -np.inf, ub_sup_x) for l_ in range(len(sample) + 1)]
    >>> ref_sample = np.ediff1d([sum(np.all((le_x < yt) & (yt < ue_x), 1)) for le_x, ue_x in edges_y])
    >>> alpha < chi2_contingency([sample, ref_sample])[1]
    True

    >>> # test against discrete approximation, bounded case
    >>> l = [sample_layer(fin_t, init_x, fin_x, lb_x, ub_sup_x) for _ in range(n)]
    >>> sample = np.bincount(l)[1:11]
    >>> yt = xt[np.all((lb_x < xt) & (xt < ub_sup_x), 1)]
    >>> edges_y = [compute_edges(l_, fin_t, init_x, fin_x, lb_x, ub_sup_x) for l_ in range(len(sample) + 1)]
    >>> ref_sample = np.ediff1d([sum(np.all((le_x < yt) & (yt < ue_x), 1)) for le_x, ue_x in edges_y])
    >>> alpha < chi2_contingency([sample, ref_sample])[1]
    True
    """

    assert 0 < fin_t

    if not np.isinf(lb_x) and not np.isinf(ub_x):
        for _ in range(max_props):
            u = ome.uniform()
            offset, width = np.mean([lb_x, ub_x]), ub_x - lb_x
            if not hitting.sample_brownbr_esc(fin_t, init_x - offset, fin_x - offset, width / 2, None, u, ome):
                break
        else:
            raise BudgetConstraintError('None of the proposals were accepted.')
    elif not np.isinf(lb_x) and np.isinf(ub_x):
        u = ome.uniform(hitting.cdf_brownbr_min(lb_x, fin_t, init_x, fin_x), 1)
    elif np.isinf(lb_x) and not np.isinf(ub_x):
        u = ome.uniform(hitting.cdf_brownbr_min(-ub_x, fin_t, -fin_x, -init_x), 1)
    else:
        u = ome.uniform()

    for l in count(1):
        edges_x = compute_edges(l, fin_t, init_x, fin_x, lb_x, ub_x)
        offset, width = np.mean(edges_x), edges_x[1] - edges_x[0]
        if not hitting.sample_brownbr_esc(fin_t, init_x - offset, fin_x - offset, width / 2, None, u, ome):
            return l


def sample_anchor(fin_t: float, init_x: float, fin_x: float, lb_x: float = -np.inf, ub_x: float = np.inf,
                  ome: np.random.Generator = np.random.default_rng()
                  ) -> (bool, Tuple[float, float], Tuple[float, float]):
    """
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param lb_x:
    :param ub_x
    :param ome:
    :return:
    """

    assert 0 < fin_t

    # sample partition
    layer_ix = sample_layer(fin_t, init_x, fin_x, lb_x, ub_x, ome)
    in_layer, ou_layer = [compute_edges(layer_ix - i, fin_t, init_x, fin_x, lb_x, ub_x) for i in (0, 1)]
    lo_sector, hi_sector = (in_layer[0], ou_layer[0]), (ou_layer[1], in_layer[1])

    # assess sector probabilities
    p_lo = hitting.cdf_brownbr_min(lo_sector[1], fin_t, init_x, fin_x, lo_sector[0])
    p_hi = hitting.cdf_brownbr_min(-hi_sector[0], fin_t, -fin_x, -init_x, -hi_sector[1])

    # simulate anchor sector
    return ome.uniform() < p_lo / (p_lo + p_hi), lo_sector, hi_sector


def sample_edges(fin_t: float, init_x: float, fin_x: float,
                 bounds_inf_x: Tuple[float, float], bounds_sup_x: Tuple[float, float],
                 ome: np.random.Generator = np.random.default_rng(),
                 max_props: int = int(1e6)) -> (float, float, float, float, Optional[List[bool]]):
    """
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param bounds_inf_x:
    :param bounds_sup_x:
    :param ome:
    :param max_props:
    :return:
    :raise: BudgetConstraintError
    """

    assert 0 < fin_t
    assert 0 < max_props

    lb_inf_x, ub_inf_x = bounds_inf_x
    lb_sup_x, ub_sup_x = bounds_sup_x

    for _ in range(max_props):
        min_t, min_x = hitting.sample_layerbr_min(fin_t, init_x, fin_x, ub_sup_x, lb_inf_x, ub_inf_x, ome)
        if min(init_x, fin_x) in (ub_inf_x, lb_sup_x):
            return min_t, min_x, lb_sup_x, ub_sup_x, None
        esc1 = hitting.sample_bessel3br_esc(min_t, 0, init_x - min_x, lb_sup_x - min_x, ub_sup_x - min_x, ome=ome)
        esc2 = hitting.sample_bessel3br_esc(fin_t - min_t, 0, fin_x - min_x, lb_sup_x - min_x, ub_sup_x - min_x, ome=ome)
        if (not esc1 and not esc2) or ome.uniform() > .5:
            return min_t, min_x, lb_sup_x, ub_sup_x, [esc1, esc2]
    else:
        raise BudgetConstraintError('None of the proposals were accepted.')


def compute_edges(i: int, fin_t: float, init_x: float, fin_x: float, lb_x: float = -np.inf, ub_x: float = np.inf,
                  pow: int = 2) -> (float, float):
    """
    :param i:
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param lb_x:
    :param ub_x:
    :param pow:
    :return:

    >>> # generate fixture
    >>> fin_t = np.random.uniform()
    >>> init_x = np.random.normal()
    >>> fin_x = np.random.normal(init_x, np.sqrt(fin_t))
    >>> lb_x = min(init_x, fin_x) - 1
    >>> ub_sup_x = max(init_x, fin_x) + 1

    >>> # test edge intitialization
    >>> np.allclose(compute_edges(0, fin_t, init_x, fin_x), (min(init_x, fin_x), max(init_x, fin_x)))
    True

    >>> # test edge convergence
    >>> np.allclose(compute_edges(int(np.nan_to_num(np.inf)), fin_t, init_x, fin_x, lb_x, ub_sup_x), (lb_x, ub_sup_x))
    True
    """

    assert 0 <= i
    assert 0 < fin_t
    assert lb_x < min(init_x, fin_x)
    assert max(init_x, fin_x) < ub_x

    if i == 0:
        return min(init_x, fin_x), max(init_x, fin_x)

    mirror = (init_x + fin_x) / 2

    le_x = hitting.ppf_brownbr_min(1 / (i ** pow + 1), fin_t, init_x, fin_x, lb_x)
    ue_x = mirror - hitting.ppf_brownbr_min(1 / (i ** pow + 1), fin_t, mirror - init_x, mirror - fin_x, mirror - ub_x)

    return le_x, ue_x
