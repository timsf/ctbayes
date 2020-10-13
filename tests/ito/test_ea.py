import unittest

import numpy as np
from scipy.stats import norm, kstest

from ctbayes.ito.models import ou
from tests.ito import fix_ou


class OuTest(unittest.TestCase):

    def setUp(self, seed=None):

        self.ome = np.random.default_rng(seed)

        self.t, self.vt, self.thi = fix_ou.generate_fixture(self.ome)
        self.xt = ou.reduce(self.thi, self.vt)
        self.fin_t = self.t[-1]
        self.init_v, self.fin_v = self.vt

    def tearDown(self):

        print('Test conducted with the following seed: {0}'.format(self.ome._bit_generator._seed_seq.entropy))

    def test_pivot(self):

        norm_rsde, denorm_rsde = ou.gen_normops(self.thi, self.t, self.vt)[0]
        test_t, test_x = denorm_rsde(*norm_rsde(self.t, self.xt))
        np.testing.assert_allclose(test_t, self.t)
        np.testing.assert_allclose(test_x, self.xt)

    def test_unconditional(self, nsamples=int(1e3), ntimes=int(1e1), alpha=1e-2):

        test_t = np.sort(self.ome.uniform(high=self.fin_t, size=ntimes))
        sampler = ou.sample_sde(self.thi, test_t, self.fin_t, self.init_v, None, self.ome)
        test_vt = np.array([next(sampler) for _ in range(nsamples)])

        true_mean, true_cov = fix_ou.eval_moments(test_t, self.init_v, None, *self.thi)
        sample = np.linalg.solve(np.linalg.cholesky(true_cov), (test_vt - true_mean).T).flatten()

        self.assertLess(alpha, kstest(sample, norm().cdf)[1])

    def test_conditional(self, nsamples=int(1e3), ntimes=int(1e1), alpha=1e-2):

        test_t = np.sort(self.ome.uniform(high=self.fin_t, size=ntimes))
        sampler = ou.sample_sde(self.thi, test_t, self.fin_t, *self.vt, self.ome)
        test_vt = np.array([next(sampler) for _ in range(nsamples)])

        true_mean, true_cov = fix_ou.eval_moments(np.append(test_t, self.fin_t), *self.vt, *self.thi)
        sample = np.linalg.solve(np.linalg.cholesky(true_cov), (test_vt - true_mean).T).flatten()

        self.assertLess(alpha, kstest(sample, norm().cdf)[1])
