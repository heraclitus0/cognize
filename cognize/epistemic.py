import numpy as np

class EpistemicState:
    """
    Cognize: Symbolic cognition layer for tracking epistemic state.
    Models projection (V), misalignment memory (E), and rupture (Θ).
    """

    def __init__(self, V0=0.0, E0=0.0, threshold=0.35, realign_strength=0.3):
        self.V = V0
        self.E = E0
        self.Θ = threshold
        self.k = realign_strength
        self.history = []

    def receive(self, R):
        """
        Accepts a signal R (reality), computes distortion ∆, and
        triggers rupture/reset if ∆ > Θ. Otherwise, updates V and E.
        """
        Δ = abs(R - self.V)
        ruptured = Δ > self.Θ

        if ruptured:
            self.V, self.E = 0.0, 0.0
        else:
            self.V += self.k * Δ * (1 + self.E)
            self.E += 0.1 * Δ

        self.history.append({'R': R, 'V': self.V, 'Δ': Δ, 'Θ': self.Θ, 'ruptured': ruptured})

    def state(self):
        """Return current epistemic state snapshot."""
        return {'V': self.V, 'E': self.E, 'Θ': self.Θ}

    def rupture_risk(self):
        """Return latest ∆ - Θ (rupture pressure)."""
        if not self.history: return None
        return self.history[-1]['Δ'] - self.Θ

    def symbol(self):
        """Return symbolic marker: ∅ (empty), ⊙ (stable), ⚠ (ruptured)."""
        if not self.history: return "∅"
        return "⚠" if self.history[-1]['ruptured'] else "⊙"
