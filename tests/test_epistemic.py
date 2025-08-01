import unittest
import numpy as np
from cognize.epistemic_state import EpistemicState
from cognize.policies import (
    collapse_soft_decay, realign_tanh, threshold_adaptive
)

class TestEpistemicStateCore(unittest.TestCase):

    def test_initial_state(self):
        state = EpistemicState()
        summary = state.summary()
        self.assertEqual(summary['V'], 0.0)
        self.assertEqual(summary['E'], 0.0)
        self.assertEqual(summary['ruptures'], 0)

    def test_receive_no_rupture(self):
        state = EpistemicState(threshold=1.0)
        state.receive(0.2)
        self.assertEqual(state.summary()['ruptures'], 0)
        self.assertFalse(state.last()['ruptured'])

    def test_receive_with_rupture(self):
        state = EpistemicState(threshold=0.1)
        state.receive(0.5)
        self.assertEqual(state.summary()['ruptures'], 1)
        self.assertTrue(state.last()['ruptured'])

    def test_manual_realign(self):
        state = EpistemicState()
        state.receive(0.3)
        state.realign(0.7)
        self.assertEqual(state.last()['event'], 'manual_realign')
        self.assertAlmostEqual(state.summary()['V'], 0.7)

    def test_reset_function(self):
        state = EpistemicState()
        state.receive(0.4)
        state.reset()
        self.assertEqual(state.summary()['V'], 0.0)
        self.assertEqual(len(state.log()), 0)

    def test_policy_injection_and_effect(self):
        state = EpistemicState()
        state.inject_policy(
            collapse=collapse_soft_decay,
            realign=realign_tanh,
            threshold=threshold_adaptive
        )
        state.receive(0.6)
        self.assertIn(state.summary()['last_event'], ['rupture', 'realign'])

    def test_drift_statistics(self):
        state = EpistemicState()
        for r in [0.1, 0.2, 0.3, 0.4]:
            state.receive(r)
        stats = state.drift_stats()
        self.assertTrue('mean_drift' in stats)
        self.assertGreater(stats['max_drift'], 0)

    def test_policy_injection_shortcut(self):
        state = EpistemicState()
        state.inject_policy(
            collapse=collapse_soft_decay,
            realign=realign_tanh,
            threshold=threshold_adaptive
        )
        state.receive(0.6)
        log = state.last()
        self.assertIsNotNone(log)
        self.assertIn(log['symbol'], ['⊙', '⚠'])

if __name__ == '__main__':
    unittest.main()
