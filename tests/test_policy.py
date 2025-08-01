import unittest
import numpy as np
from cognize.policies import (
    collapse_reset,
    collapse_soft_decay,
    collapse_adopt_R,
    collapse_randomized,
    realign_linear,
    realign_tanh,
    realign_bounded,
    realign_decay_adaptive,
    threshold_static,
    threshold_adaptive,
    threshold_stochastic,
    threshold_combined
)

class TestPolicyFunctions(unittest.TestCase):

    def test_collapse_reset(self):
        V, E = collapse_reset(R=0.5, V=0.3, E=0.7)
        self.assertEqual(V, 0.0)
        self.assertEqual(E, 0.0)

    def test_collapse_soft_decay(self):
        V, E = collapse_soft_decay(R=0.5, V=1.0, E=0.5)
        self.assertAlmostEqual(V, 0.5)
        self.assertAlmostEqual(E, 0.15)

    def test_collapse_adopt_R(self):
        V, E = collapse_adopt_R(R=0.9, V=0.1, E=1.0)
        self.assertEqual(V, 0.9)
        self.assertEqual(E, 0.0)

    def test_collapse_randomized(self):
        V, E = collapse_randomized(R=0.0, V=0.0, E=0.0)
        self.assertIsInstance(V, float)
        self.assertEqual(E, 0.0)

    def test_realign_linear(self):
        V_new = realign_linear(V=0.2, delta=0.4, E=0.5, k=0.3)
        self.assertGreater(V_new, 0.2)

    def test_realign_tanh(self):
        V_new = realign_tanh(V=0.1, delta=1.0, E=0.5, k=1.0)
        self.assertTrue(0.1 < V_new < 1.5)

    def test_realign_bounded(self):
        V_new = realign_bounded(V=0.0, delta=10, E=1.0, k=1.0, cap=0.5)
        self.assertLessEqual(V_new, 0.5)

    def test_realign_decay_adaptive(self):
        V_new = realign_decay_adaptive(V=0.1, delta=0.5, E=2.0, k=1.0)
        self.assertGreater(V_new, 0.1)

    def test_threshold_static(self):
        Θ = threshold_static(E=0.0, t=0)
        self.assertAlmostEqual(Θ, 0.35)

    def test_threshold_adaptive(self):
        Θ = threshold_adaptive(E=2.0, t=10)
        self.assertGreater(Θ, 0.35)

    def test_threshold_stochastic(self):
        Θ = threshold_stochastic(E=1.0, t=5)
        self.assertIsInstance(Θ, float)

    def test_threshold_combined(self):
        Θ = threshold_combined(E=1.5, t=7)
        self.assertGreater(Θ, 0.35)

if __name__ == '__main__':
    unittest.main()
