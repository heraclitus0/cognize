# Unit tests for custom user-defined policies injected into EpistemicState

import unittest
from cognize.epistemic_state import EpistemicState

class TestCustomLogic(unittest.TestCase):

    def test_custom_threshold_function(self):
        def custom_threshold(E, t):
            return 0.2 if t < 5 else 0.5

        agent = EpistemicState(threshold_fn=custom_threshold)
        for t in range(10):
            agent.receive(0.4)
        rupture_count = agent.summary()['ruptures']
        self.assertGreater(rupture_count, 0)

    def test_custom_collapse_function(self):
        def custom_collapse(R, V, E):
            return (R + V) / 2, E * 0.1

        agent = EpistemicState(threshold=0.1, collapse_fn=custom_collapse)
        agent.receive(1.0)
        v_post = agent.summary()['V']
        self.assertNotEqual(v_post, 0.0)
        self.assertLess(agent.summary()['E'], 0.1)

    def test_custom_realign_function(self):
        def custom_realign(V, delta, E, k):
            return V + delta**2 - 0.1 * E

        agent = EpistemicState(realign_fn=custom_realign)
        agent.receive(0.5)
        v_updated = agent.summary()['V']
        self.assertGreater(v_updated, 0.0)

    def test_runtime_policy_injection(self):
        agent = EpistemicState()

        def dynamic_threshold(E, t): return 0.1 + 0.1 * E
        def bounded_collapse(R, V, E): return max(min(R, 1.0), -1.0), E * 0.5
        def fast_realign(V, delta, E, k): return V + 2 * delta

        agent.inject_policy(
            threshold=dynamic_threshold,
            collapse=bounded_collapse,
            realign=fast_realign
        )

        for r in [0.1, 0.5, 0.9]:
            agent.receive(r)
        self.assertTrue(agent.summary()['ruptures'] >= 0)
        self.assertIn('event', agent.last())

if __name__ == '__main__':
    unittest.main()
