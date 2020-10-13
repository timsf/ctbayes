import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import ttest_1samp

from ctbayes.mjplib import skeleton as mjp_skel, inference as mjp_inf
from ctbayes.switching import mcem
from ctbayes.switching.models import ou

from tests.switching import fix_ou


def eval_map(t, vt, init_thi, init_lam, bounds_thi, hyper_lam, f_loglik, f_logprior, ome):

    return iterate_em(t, vt, init_thi, init_lam, -np.inf, bounds_thi, hyper_lam, f_loglik, f_logprior, 8, ome)


def iterate_em(t, vt, thi, lam, obj, bounds_thi, hyper_lam, f_loglik, f_logprior, n_particles, ome, ftol=.01):

    yy = update_hidden(t[-1], lam, n_particles, ome)
    thi_prime, obj_thi = update_param(t, vt, yy, thi, bounds_thi, f_loglik, f_logprior)
    lam_prime, obj_lam = update_generator(yy, hyper_lam)
    obj_prime = obj_thi + obj_lam
    if np.abs(obj_prime - obj) < ftol:
        return thi_prime, lam_prime
    else:
        return iterate_em(t, vt, thi_prime, lam_prime, obj_prime, bounds_thi, hyper_lam,
                          f_loglik, f_logprior, n_particles + 2, ome, ftol)


def update_hidden(fin_t, lam, n_particles, ome):

    return [mjp_skel.sample_forwards(fin_t, None, lam, ome) for _ in range(n_particles)]


def update_param(t, vt, y, thi_nil, bounds_thi, f_loglik, f_logprior):

    def eval_obj(thi_flat):
        thi_ = np.reshape(thi_flat, thi_nil.shape)
        return -(np.mean([np.exp(log_w_) * f_loglik(t, vt, y_, *thi_.T) for log_w_, y_ in zip(log_w, y)]) + f_logprior(thi_))

    log_w = np.array([f_loglik(t, vt, y_, *thi_nil.T) for y_ in y])
    log_w = log_w - logsumexp(log_w) + np.log(len(log_w))
    d_thi = np.repeat([min(abs(lb_ - ub_) / 2, 1e-2) for lb_, ub_ in zip(*bounds_thi)], thi_nil.shape[0])
    res = minimize(lambda thi_: eval_obj(thi_),
                   thi_nil, bounds=list(zip(np.repeat(bounds_thi[0], 2) + d_thi, np.repeat(bounds_thi[1], 2) - d_thi)))
    thi_prime = np.reshape(res.x, thi_nil.shape)
    obj_prime = -res.fun
    return thi_prime, obj_prime


def update_generator(yy, hyper_lam) -> (np.ndarray, float):

    alp, bet = zip(*[mjp_inf.update(y, *hyper_lam) for y in yy])
    lam_prime = np.mean([(a - 1) for a in alp], 0) / np.mean([b for b in bet], 0)
    lam_prime[np.isnan(lam_prime)] = -np.nansum(lam_prime, 1)
    log_post = np.mean([mjp_inf.eval_loglik(y, lam_prime) for y in yy]) + mjp_inf.eval_logprior(lam_prime, *hyper_lam)
    return lam_prime, log_post


class OuTest(unittest.TestCase):

    def setUp(self, seed=None):

        self.ome = np.random.default_rng(seed)

        self.t, self.vt, self.y, self.thi, self.lam, self.lam0 = fix_ou.generate_fixture(self.ome)
        self.xt = ou.reduce(self.vt)
        self.mod = mcem.Model(self.t, self.vt, ou.bounds_thi, self.lam0, ou.eval_log_prior, ou.sample_aug, ou.gen_normops,
                              ou.eval_biased_log_lik, ou.eval_disc, ou.eval_bounds_disc)

    def tearDown(self):

        print('Test conducted with the following seed: {0}'.format(self.ome._bit_generator._seed_seq.entropy))

    def test_pivot(self):

        _, h = ou.sample_aug(self.thi, self.y, self.t, self.vt, self.ome)
        norm_rsde, denorm_rsde = ou.gen_normops(self.thi, *h)[0]
        test_t, test_x = denorm_rsde(*norm_rsde(self.t, self.xt))
        np.testing.assert_allclose(test_t, self.t)
        np.testing.assert_allclose(test_x, self.xt)

    def test_trans_est(self, nsamples=int(1e4), alpha=1e-2):

        z, h = zip(*[ou.sample_aug(self.thi, self.y, self.t, self.vt, self.ome) for _ in range(nsamples)])
        est_log_aug_p, z = zip(*[mcem.est_aug_lik(self.thi, z_, h_, self.mod, self.ome) for z_, h_ in zip(z, h)])
        est_aug_p = np.exp(est_log_aug_p)
        true_p = np.exp(fix_ou.eval_log_lik(self.t, self.vt, self.y, *self.thi.T))
        self.assertLess(alpha, ttest_1samp(est_aug_p, true_p)[1])

    def test_map_est(self, tol=1e-1):

        estimator = ou.maximize_posterior(self.t, self.vt, 2, ome=self.ome)
        obj, est_map_thi, est_map_lam = list(estimator)[-1][:3]
        true_map_thi, true_map_lam = eval_map(self.t, self.vt, self.thi, self.lam,
                                              self.mod.bounds_thi, self.mod.hyper_lam,
                                              fix_ou.eval_log_lik, self.mod.eval_log_prior, self.ome)
        np.testing.assert_allclose(est_map_thi, true_map_thi, tol)
        np.testing.assert_allclose(est_map_lam, true_map_lam, tol)
