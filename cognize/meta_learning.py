from typing import Dict, List, Callable, Optional
import copy
import numpy as np

class PolicySpec:
    def __init__(self, id: str, threshold_fn: Callable, realign_fn: Callable, collapse_fn: Callable, params: Dict):
        self.id, self.threshold_fn, self.realign_fn, self.collapse_fn, self.params = id, threshold_fn, realign_fn, collapse_fn, params
    def fns(self): return {"threshold": self.threshold_fn, "realign": self.realign_fn, "collapse": self.collapse_fn}

class PolicyMemory:
    def __init__(self, cap: int = 5000):
        self.records: List[Dict] = []; self.cap = cap
    def remember(self, ctx: Dict, spec: PolicySpec, reward: float):
        self.records.append({"ctx": ctx, "id": spec.id, "params": spec.params, "reward": float(reward)})
        if len(self.records) > self.cap: self.records = self.records[-self.cap:]

class ShadowRunner:
    def replay(self, state, spec: PolicySpec, evidence_seq) -> float:
        # Clone-ish: shallow copy & reset logs
        s = copy.deepcopy(state)
        s.inject_policy(**spec.fns())
        base_t = len(s.history)
        for r in evidence_seq: s.receive(r, source="shadow")
        # Simple reward: low rupture rate, low mean drift, fast realign
        window = s.history[base_t:]
        if not window: return -1.0
        drift = np.array([w["∆"] for w in window], float)
        rupt = np.array([w["ruptured"] for w in window], bool).mean()
        reward = - rupt - drift.mean()
        return float(reward)

class PolicyManager:
    def __init__(self, base_specs: List[PolicySpec], memory: PolicyMemory, shadow: ShadowRunner, epsilon: float = 0.1):
        self.specs, self.memory, self.shadow, self.epsilon = base_specs, memory, shadow, float(epsilon)
        self.last_promotion = None

    def maybe_adjust(self, state, ctx: Dict, recent_evidence):
        # ε-greedy exploration
        explore = np.random.random() < self.epsilon
        candidates = self.specs if explore else self._top_k(ctx, k=3)
        best_reward, best_spec = -1e9, None
        for spec in candidates:
            r = self.shadow.replay(state, spec, recent_evidence)
            if r > best_reward: best_reward, best_spec = r, spec
        # Promote if it beats current behavior by margin
        current_reward = self.shadow.replay(state, self._current_as_spec(state), recent_evidence)
        if best_spec and best_reward > current_reward * 1.05:
            state.inject_policy(**best_spec.fns())
            self.memory.remember(ctx, best_spec, best_reward)
            self.last_promotion = {"id": best_spec.id, "reward": best_reward}

    def _top_k(self, ctx, k=3):
        # naive: pick globally best k; later, bucket by ctx
        recs = sorted(self.memory.records, key=lambda r: r["reward"], reverse=True)
        ids = []
        for r in recs:
            if r["id"] not in ids: ids.append(r["id"])
            if len(ids) >= k: break
        by_id = {s.id: s for s in self.specs}
        return [by_id[i] for i in ids] if ids else self.specs

    def _current_as_spec(self, state):
        # Snapshot current policies as a temporary spec for comparison
        return PolicySpec("current", state._threshold_fn or (lambda s: state.Θ),
                          state._realign_fn or (lambda s, R, d: state.V + state.k*d),
                          state._collapse_fn or (lambda s, R: (0.0, 0.0)), params={})
