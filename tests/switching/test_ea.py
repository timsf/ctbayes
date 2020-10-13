import unittest

import numpy as np

from ctbayes.switching.models import ou
from tests.switching import fix_ou


class OuTest(unittest.TestCase):

    def setUp(self, seed=None):

        self.ome = np.random.default_rng(seed)

        self.t, self.vt, self.y, self.thi, self.lam, self.lam0 = fix_ou.generate_fixture(self.ome)
        self.xt = ou.reduce(self.vt)

    def tearDown(self):

        print('Test conducted with the following seed: {0}'.format(self.ome._bit_generator._seed_seq.entropy))

    def test_pivot(self):

        _, h = ou.sample_aug(self.thi, self.y, self.t, self.vt, self.ome)
        norm_rsde, denorm_rsde = ou.gen_normops(self.thi, *h)[0]
        test_t, test_x = denorm_rsde(*norm_rsde(self.t, self.xt))
        np.testing.assert_allclose(test_t, self.t)
        np.testing.assert_allclose(test_x, self.xt)
