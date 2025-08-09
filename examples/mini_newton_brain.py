
# Requires: pip install cognize matplotlib

import matplotlib.pyplot as plt
from datetime import datetime
from cognize import EpistemicState

# --- 1) Conceptual policies (domain logic) ---
def collapse_slow(belief, E):
    """Small decay toward conservative prior, memory softens too."""
    return belief * 0.98, E * 0.90  # MUST return (V, E)

def realign_gravity(belief, evidence, E):
    """Move belief toward evidence; sensitivity increases with misalignment memory E."""
    alpha = 0.15 * (1 + 0.5 * min(E, 1.0))
    return belief + alpha * (evidence - belief)

def adaptive_threshold(E, base=0.4):
    """Conservative when memory is low; more sensitive after repeated mismatch."""
    return base * (1 - 0.3 * min(E, 1.0))

# --- 2) Adapters to Cognize’s expected signatures ---
# EpistemicState will call:
#   threshold_fn(self) -> float Θ
#   realign_fn(self, R, delta) -> new V
#   collapse_fn(self) -> (new V, new E)
collapse_slow_fn       = lambda self: collapse_slow(self.V, self.E)
realign_gravity_fn     = lambda self, R, d: realign_gravity(self.V, R, self.E)
adaptive_threshold_fn  = lambda self: adaptive_threshold(self.E)

# --- 3) Create the epistemic agent (Newton brain) ---
newton = EpistemicState(V0=0.2, threshold=0.4)  # initial weak belief
newton.inject_policy(
    collapse=collapse_slow_fn,
    realign=realign_gravity_fn,
    threshold=adaptive_threshold_fn
)

# --- 4) Sequence of observations (event, evidence in [0,1]) ---
observations = [
    ("apple_fall_small",   0.25),
    ("apple_fall_repeat",  0.30),
    ("cannonball_exp",     0.45),
    ("moon_motion_hint",   0.55),
    ("tide_correlation",   0.85),  # likely rupture
    ("planet_orbit_calc",  0.92)   # consolidation
]

# --- 5) Run simulation & collect trace using Cognize’s log for ground truth ---
trace = []
for t, (evt, evidence) in enumerate(observations, start=1):
    before = newton.V
    newton.receive(evidence, source=evt)
    last = newton.last() or {}
    row = {
        "t": t,
        "event": evt,
        "evidence": float(last.get("R", evidence)),
        "belief_before": float(before),
        "belief_after": float(last.get("V", newton.V)),
        "misalignment": float(last.get("∆", abs(evidence - before))),
        "threshold": float(last.get("Θ", newton.Θ)),   # NOTE: property is Θ, not 'threshold'
        "E": float(getattr(newton, "E", 0.0)),
        "rupture": bool(last.get("ruptured", False)),
        "symbol": last.get("symbol", "⊙"),
        "ts": datetime.utcnow().isoformat()
    }
    trace.append(row)

# --- 6) Print a readable log ---
print("\n--- Mini-Newton Brain Log ---")
for r in trace:
    marker = "RUPTURE" if r["rupture"] else "•"
    print(
        f"[t={r['t']}] {r['event']}: "
        f"R={r['evidence']:.2f}, "
        f"V {r['belief_before']:.2f} → {r['belief_after']:.2f}, "
        f"∆={r['misalignment']:.2f}, Θ={r['threshold']:.2f} {marker}"
    )

# --- 7) Plot belief vs evidence with rupture markers ---
times     = [r["t"] for r in trace]
beliefs   = [r["belief_after"] for r in trace]
evidences = [r["evidence"] for r in trace]
rupts     = [r["t"] for r in trace if r["rupture"]]

plt.figure(figsize=(9, 4.5))
plt.plot(times, beliefs,   marker="o", label="Belief V")
plt.plot(times, evidences, marker="s", label="Evidence R")
for rt in rupts:
    plt.axvline(rt, linestyle="--", alpha=0.5)
plt.xticks(times, [r["event"] for r in trace], rotation=20)
plt.ylim(-0.05, 1.05)
plt.ylabel("Value (0..1)")
plt.title("Mini-Newton Brain: Belief vs Evidence — rupture lines marked")
plt.legend()
plt.tight_layout()
plt.show()
