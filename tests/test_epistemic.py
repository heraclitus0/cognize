
import unittest
import numpy as np
import os
from cognize.epistemic import EpistemicState

class TestEpistemicState(unittest.TestCase):

    def test_initialization(self):
        e = EpistemicState()
        self.assertEqual(e.V, 0.0)
        self.assertEqual(e.E, 0.0)
        self.assertEqual(e._rupture_count, 0)

    def test_receive_no_rupture(self):
        e = EpistemicState(V0=0.0, threshold=1.0)
        e.receive(0.2)
        self.assertFalse(e.last()["ruptured"])
        self.assertEqual(e.symbol(), "⊙")

    def test_receive_rupture(self):
        e = EpistemicState(V0=0.0, threshold=0.1)
        e.receive(1.0)
        self.assertTrue(e.last()["ruptured"])
        self.assertEqual(e.V, 0.0)
        self.assertEqual(e.E, 0.0)

    def test_reset(self):
        e = EpistemicState()
        e.receive(1.0)
        e.reset()
        self.assertEqual(e.V, 0.0)
        self.assertEqual(e.E, 0.0)
        self.assertEqual(e.history, [])

    def test_realignment(self):
        e = EpistemicState()
        e.realign(2.0)
        self.assertEqual(e.V, 2.0)

    def test_drift_stats(self):
        e = EpistemicState(V0=0.0, threshold=10.0)
        for r in [0.5, 1.0, 1.5]:
            e.receive(r)
        stats = e.drift_stats(window=3)
        self.assertIn("mean_drift", stats)

    def test_export_json_csv(self):
        e = EpistemicState()
        e.receive(1.0)

        e.export_json("test_log.json")
        self.assertTrue(os.path.exists("test_log.json"))

        e.export_csv("test_log.csv")
        self.assertTrue(os.path.exists("test_log.csv"))

        os.remove("test_log.json")
        os.remove("test_log.csv")

    def test_event_log(self):
        e = EpistemicState()
        e._log_event("test_event", {"foo": "bar"})
        logs = e.event_log_summary()
        self.assertTrue(any(ev["event"] == "test_event" for ev in logs))

    def test_vector_input(self):
        e = EpistemicState()
        e.receive([1.0, 2.0, 3.0])
        self.assertIsInstance(e.last(), dict)
        self.assertIn("delta", e.last())

if __name__ == "__main__":
    unittest.main()
