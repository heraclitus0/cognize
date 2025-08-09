from typing import Dict, Set
import numpy as np
from .epistemic import EpistemicState

class EpistemicGraph:
    def __init__(self, damping: float = 0.5):
        self.nodes: Dict[str, EpistemicState] = {}
        self.edges: Dict[str, Set[str]] = {}
        self.damping = float(damping)

    def add(self, name: str, **kwargs):
        self.nodes[name] = EpistemicState(**kwargs); self.edges[name] = set()

    def link(self, src: str, dst: str):
        self.edges[src].add(dst)

    def step(self, name: str, R):
        n = self.nodes[name]
        pre = n.last()
        n.receive(R, source=name)
        post = n.last()
        if post and post["ruptured"]:
            pressure = post["∆"] - post["Θ"]
            for dst in self.edges[name]:
                m = self.nodes[dst]
                influence = np.clip(self.damping * pressure, -1.0, 1.0)
                # Push neighbor toward its own R by nudging V (soft collapse-like)
                m.V = m.V + np.sign((post["R"] - float(m.V))) * influence
