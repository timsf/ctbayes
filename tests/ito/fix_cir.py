import numpy as np

from scipy.stats import ncx2


def eval_log_lik(t, vt, bet=1, mu=1, sig=1):

    c = 2 * bet / (sig ** 2 * (1 - np.exp(-bet * np.diff(t))))
    zt = 2 * c * vt[1:]
    nc = 2 * np.exp(-bet * np.diff(t)) * c * vt[:-1]
    df = 4 * bet * mu / sig ** 2
    
    return (ncx2(df, nc).logpdf(zt) + np.log(2 * c)).sum()


def sample_path(t, init_x, bet, mu, sig, ome=np.random.default_rng()):
    """Sample from a CIR process at given times.

    :param t:
    :param init_x:
    :param bet:
    :param mu:
    :param sig:
    :param ome:
    :return:

    >>> # generate fixture
    >>> n = int(1e4)
    >>> bet, mu, sig = np.random.lognormal(size=3)
    >>> init_x = np.random.lognormal()
    >>> t = np.random.uniform(size=1)

    >>> # draw iid samples
    >>> xt = np.array([sample_path(t, init_x, bet, mu, sig) for _ in range(n)])

    >>> # test marginal sampling distribution
    >>> from scipy.stats import kstest, ncx2
    >>> alpha = 1e-2
    >>> sample = (4 * bet / (sig ** 2 * (1 - np.exp(-bet * t))) * xt).flatten()
    >>> dist = ncx2(4 * bet * mu / sig ** 2, 4 * bet / (sig ** 2 * (np.exp(bet * t) - 1)) * init_x)
    >>> alpha < kstest(sample, dist.cdf)[1]
    True
    """

    assert 0 < np.min(t)
    assert 0 <= np.min(init_x)
    assert 0 < bet
    assert 0 < mu
    assert 0 < sig

    df = 4 * bet * mu / sig ** 2
    xt = [init_x]
    for dt in np.diff(np.hstack([0, t])):
        c = 4 * bet / (sig ** 2 * (1 - np.exp(-bet * dt)))
        xt.append(ome.noncentral_chisquare(df, c * np.exp(-bet * dt) * xt[-1]) / c)

    return np.array(xt[1:])


def generate_fixture(ome):

    bet, mu = ome.exponential(size=2)
    sig = ome.uniform(np.sqrt(bet * mu * 4 / 3) / 2, np.sqrt(bet * mu * 4 / 3))
    fin_t = ome.uniform()
    init_v = ome.gamma(2 * bet * mu / sig ** 2, sig ** 2 / (2 * bet))
    fin_v, = sample_path(np.array([fin_t]), init_v, bet, mu, sig, ome)

    t = np.array([0, fin_t])
    vt = np.array([init_v, fin_v])
    thi = np.array([bet, mu, sig])
    
    return t, vt, thi
