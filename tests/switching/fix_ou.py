import numpy as np

from scipy.stats import norm

from ctbayes.mjplib import skeleton
from ctbayes.mjplib.skeleton import sample_forwards, sample_trial_generator, get_stat
from tests.ito import fix_ou as fix_ito_ou


def eval_log_lik(t, vt, y, bet, sig):

    if np.any(bet <= 0) or np.any(sig <= 0):
        return -np.inf
    return sum([eval_log_trans(dt, v0, v1, y_, bet, sig)
                for dt, v0, v1, y_ in zip(np.diff(t), vt, vt[1:], skeleton.partition_skeleton(y, t[1:-1]))])


def sample_path(t, init_v, y, bet, sig, ome):

    vt = [init_v]
    for y_ in skeleton.partition_skeleton(y, t[:-1]):
        vt.append(sample_trans(vt[-1], y_, bet, sig, ome))
    return np.array(vt[1:])


def eval_moments(t, init_v, fin_v, y, bet, sig):

    insert_to = np.searchsorted(np.append(y.t, y.fin_t), t[:-1])
    full_y = skeleton.Skeleton(
        np.insert(np.append(y.t, y.fin_t), insert_to, t[:-1])[:-1],
        np.insert(np.append(y.xt, y.xt[-1]), insert_to, y.xt[insert_to - 1])[:-1],
        y.fin_t)

    full_mean, full_cov = eval_moments_trans(init_v, full_y, bet, sig)

    mean = full_mean[np.append(insert_to - 1, -1)]
    cov = full_cov[np.append(insert_to - 1, -1)][:, np.append(insert_to - 1, -1)]

    if fin_v is None:
        return mean, cov
    else:
        return condition(mean, cov, fin_v)


def eval_log_trans(fin_t, init_v, fin_v, y, bet, sig):

    if np.any(bet <= 0) or np.any(sig <= 0):
        return -np.inf
    mean, cov = eval_moments(np.array([fin_t]), init_v, None, y, bet, sig)
    return norm(mean[0], np.sqrt(cov[0, 0])).logpdf(fin_v)


def sample_trans(init_v, y, bet, sig, ome):

    vt = [init_v]
    for dt, z0 in zip(np.diff(np.append(y.t, y.fin_t)), y.xt):
        vt.extend(fix_ito_ou.sample_path(np.array([dt]), vt[-1], bet[z0], sig[z0], ome))
    return vt[-1]


def eval_moments_trans(init_v, y, bet, sig):

    coefs, vars = eval_coefs(y, bet, sig)
    cum_vars = [np.append(np.cumprod(np.square(coefs[1:i + 1])[::-1])[::-1], 1) @ vars[:i + 1] for i in range(len(coefs))]

    mean = np.cumprod(coefs) * init_v
    cov = np.empty(2 * y.t.shape)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):
            cov[i, j] = np.prod(coefs[min(i, j) + 1:max(i, j) + 1]) * cum_vars[min(i, j)]

    return mean, cov


def eval_coefs(y, bet, sig):

    dt = np.diff(np.append(y.t, y.fin_t))
    coefs = np.exp(-bet[y.xt] * dt)
    vars = (1 - np.square(coefs)) * np.square(sig[y.xt]) / (2 * bet[y.xt])
    return coefs, vars


def condition(mean, cov, fin_v):

    cmean = mean[:-1] + cov[-1, :-1] * (fin_v - mean[-1]) / cov[-1, -1]
    ccov = cov[:-1, :-1] - np.outer(cov[-1, :-1], cov[-1, :-1]) / cov[-1, -1]
    return cmean, ccov


def generate_fixture(ome):

    bet = ome.exponential(size=2)
    sig = np.sqrt(1 / ome.gamma(2, size=2))
    lam = 10 * sample_trial_generator(2)

    fin_t = ome.uniform()
    init_y = np.argmax(ome.multinomial(1, get_stat(lam)))
    init_v = ome.normal(scale=sig[init_y] / np.sqrt(2 * bet[init_y]))

    y = sample_forwards(fin_t, init_y, lam, ome)
    fin_v, = sample_path(np.array([fin_t]), init_v, y, bet, sig, ome)

    vt = np.array([init_v, fin_v])
    t = np.array([0, y.fin_t])
    thi = np.array([bet, sig]).T

    alp0 = np.ones((2, 2)) + np.diag(np.repeat(np.nan, 2))
    bet0 = np.ones((2, 2)) + np.diag(np.repeat(np.nan, 2))
    lam0 = (2 * alp0, 2 * bet0)

    return t, vt, y, thi, lam, lam0
