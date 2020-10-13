import numpy as np

from scipy.stats import norm

from ctbayes.sdelib.paths import sample_wiener
from ctbayes.sdelib.moments import moments_wiener


def eval_log_lik(t, vt, bet, sig):

    if np.any(bet <= 0) or np.any(sig <= 0):
        return -np.inf
    return sum([eval_log_trans(dt, v0, v1, bet, sig) for dt, v0, v1 in zip(np.diff(t), vt, vt[1:])])


def sample_path(t, init_v, bet, sig, ome=np.random.default_rng()):
    """Sample from an Ornstein Uhlenbeck process at given times.

    :param t:
    :param init_v:
    :param bet:
    :param sig:
    :return:

    >>> # generate fixture
    >>> n = int(1e4)
    >>> init_x = np.random.normal()
    >>> bet = np.random.lognormal()
    >>> sig = np.random.lognormal()
    >>> t = np.sort(np.random.uniform(size=10))

    >>> # draw iid samples
    >>> vt = np.array([sample_path(t, init_v, bet, sig) for _ in range(n)])

    >>> # test mahalanobis distance sampling distribution
    >>> from scipy.stats import kstest, norm
    >>> alpha = 1e-2
    >>> true_mean, true_cov = eval_moments(t, init_v, bet, sig)
    >>> sample = np.linalg.solve(np.linalg.cholesky(true_cov), (vt - true_mean).T).flatten()
    >>> alpha < kstest(sample, norm().cdf)[1]
    True
    """

    assert 0 < np.min(t)
    assert 0 < bet
    assert 0 < sig

    wt, = sample_wiener(np.exp(2 * bet * t) - 1, np.zeros(1), ome=ome).T
    vt = init_v * np.exp(-bet * t) + wt * sig * np.exp(-bet * t) / np.sqrt(2 * bet)

    return vt


def eval_moments(t, init_v, fin_v=None, bet=1, sig=1):

    assert 0 < np.min(t)
    assert 0 < bet
    assert 0 < sig

    if np.allclose(bet, 0):
        mean, cov = moments_wiener(t, init_v)
        cov *= sig ** 2
    else:
        tmat = np.meshgrid(t, t)
        mean = init_v * np.exp(-bet * t)
        cov = sig ** 2 / (2 * bet) * (np.exp(-bet * np.abs(np.diff(tmat, axis=0)[0])) - np.exp(-bet * np.sum(tmat, 0)))

    if fin_v is None:
        return mean, cov
    else:
        return condition(mean, cov, fin_v)


def eval_log_trans(fin_t, init_v, fin_v, bet, sig):

    if bet <= 0 or sig <= 0:
        return -np.inf
    mean, cov = eval_moments(np.array([fin_t]), init_v, None, bet, sig)
    return norm(mean[0], np.sqrt(cov[0, 0])).logpdf(fin_v)


def condition(mean, cov, fin_v):

    cmean = mean[:-1] + cov[-1, :-1] * (fin_v - mean[-1]) / cov[-1, -1]
    ccov = cov[:-1, :-1] - np.outer(cov[-1, :-1], cov[-1, :-1]) / cov[-1, -1]
    return cmean, ccov


def generate_fixture(ome):

    bet = ome.exponential()
    sig = np.sqrt(1 / ome.gamma(2))
    fin_t = ome.uniform()
    init_v = ome.normal(scale=sig / np.sqrt(2 * bet))
    fin_v, = sample_path(np.array([fin_t]), init_v, bet, sig, ome)

    t = np.array([0, fin_t])
    vt = np.array([init_v, fin_v])
    thi = np.array([bet, sig])

    return t, vt, thi
