import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.stats import ttest_1samp

from ctbayes.ito import mcem
from ctbayes.ito.models import cir, ou

from tests.ito import fix_cir, fix_ou


def eval_map(t, vt, init_thi, bounds_thi, f_loglik, f_logprior):

    d_thi = np.array([min(abs(lb_ - ub_) / 2, 1e-2) for lb_, ub_ in zip(*bounds_thi)])
    res = minimize(lambda thi: -(f_loglik(t, vt, *thi) + f_logprior(thi)),
                   init_thi, bounds=list(zip(bounds_thi[0] + d_thi, bounds_thi[1] - d_thi)))

    return res.x


class OuTest(unittest.TestCase):

    def setUp(self, seed=None):

        self.ome = np.random.default_rng(seed)

        self.t, self.vt, self.thi = fix_ou.generate_fixture(self.ome)
        self.xt = ou.reduce(self.thi, self.vt)
        self.mod = mcem.Model(self.t, self.vt, ou.bounds_thi, ou.eval_log_prior, ou.sample_aug, ou.gen_normops,
                              ou.eval_biased_log_lik, ou.eval_disc, ou.eval_bounds_disc)

    def tearDown(self):

        print('Test conducted with the following seed: {0}'.format(self.ome._bit_generator._seed_seq.entropy))

    def test_trans_est(self, nsamples=int(1e3), alpha=1e-2):

        z = [ou.sample_aug(self.thi, self.t, self.vt, self.ome) for _ in range(nsamples)]
        est_log_aug_p, z = zip(*[mcem.est_aug_lik(self.thi, z_, self.mod, self.ome) for z_ in z])
        est_aug_p = np.exp(est_log_aug_p)
        true_p = np.exp(fix_ou.eval_log_lik(self.t, self.vt, *self.thi))
        self.assertLess(alpha, ttest_1samp(est_aug_p, true_p)[1])

    def test_log_trans_est(self, nsamples=int(1e3), alpha=1e-2):

        supp_t = [[np.sort(self.ome.uniform(high=dt, size=1)) for dt in np.diff(self.t)] for _ in range(nsamples)]
        z = [ou.sample_aug(self.thi, self.t, self.vt, self.ome) for _ in range(nsamples)]
        est_log_aug_p, z = zip(*[mcem.est_log_aug_lik(self.thi, z_, self.mod, t_, self.ome) for z_, t_ in zip(z, supp_t)])
        true_log_p = fix_ou.eval_log_lik(self.t, self.vt, *self.thi)
        self.assertLess(alpha, ttest_1samp(est_log_aug_p, true_log_p)[1])

    def test_map_est(self, tol=1e-1):

        estimator = ou.maximize_posterior(self.t, self.vt, 2, ome=self.ome)
        obj, est_map = list(estimator)[-1][:2]
        true_map = eval_map(self.t, self.vt, ou.init_thi, ou.bounds_thi, fix_ou.eval_log_lik, ou.eval_log_prior)
        np.testing.assert_allclose(est_map, true_map, tol)


class CirTest(unittest.TestCase):

    def setUp(self, seed=None):

        self.ome = np.random.default_rng(seed)

        self.t, self.vt, self.thi = fix_cir.generate_fixture(self.ome)
        self.xt = cir.reduce(self.thi, self.vt)
        self.mod = mcem.Model(self.t, self.vt, cir.bounds_thi, cir.eval_log_prior, cir.sample_aug, cir.gen_normops,
                              cir.eval_biased_log_lik, cir.eval_disc, cir.eval_bounds_disc)

    def tearDown(self):

        print('Test conducted with the following seed: {0}'.format(self.ome._bit_generator._seed_seq.entropy))

    def test_pivot(self):

        norm_rsde, denorm_rsde = cir.gen_normops(self.thi, self.t, self.vt)[0]
        test_t, test_x = denorm_rsde(*norm_rsde(self.t, self.xt))
        np.testing.assert_allclose(test_t, self.t)
        np.testing.assert_allclose(test_x, self.xt)

    def test_trans_est(self, nsamples=int(1e4), alpha=1e-2):

        z = [cir.sample_aug(self.thi, self.t, self.vt, self.ome) for _ in range(nsamples)]
        est_log_aug_p, z = zip(*[mcem.est_aug_lik(self.thi, z_, self.mod, self.ome) for z_ in z])
        est_aug_p = np.exp(est_log_aug_p)
        true_dens = np.exp(fix_cir.eval_log_lik(self.t, self.vt, *self.thi))
        self.assertLess(alpha, ttest_1samp(est_aug_p, true_dens)[1])

    def test_log_trans_est(self, nsamples=int(1e4), alpha=1e-2):

        supp_t = [[np.sort(self.ome.uniform(high=dt, size=1)) for dt in np.diff(self.t)] for _ in range(nsamples)]
        z = [cir.sample_aug(self.thi, self.t, self.vt, self.ome) for _ in range(nsamples)]
        est_log_aug_p, z = zip(*[mcem.est_log_aug_lik(self.thi, z_, self.mod, t_, self.ome) for z_, t_ in zip(z, supp_t)])
        true_log_p = fix_cir.eval_log_lik(self.t, self.vt, *self.thi)
        self.assertLess(alpha, ttest_1samp(est_log_aug_p, true_log_p)[1])
