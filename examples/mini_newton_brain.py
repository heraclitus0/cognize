import matplotlib.pyplot as plt
from datetime import datetime, UTC
from cognize import EpistemicState

# 1) Conceptual scalar policies
def collapse_slow(belief, E):
    """Gentle reset: decay belief toward prior; soften memory."""
    return belief * 0.98, E * 0.90  # MUST return (V, E)

def realign_gravity(belief, evidence, E):
    """Pull belief toward evidence; higher E => stronger pull."""
    alpha = 0.15 * (1 + 0.5 * min(E, 1.0))
    return belief + alpha * (evidence - belief)

def adaptive_threshold(E, base=0.40):
    """Lower Θ when memory is high (more sensitive after repeated mismatch)."""
    return base * (1 - 0.30 * min(E, 1.0))

# 2) Adapt to Cognize’s expected signatures
#    threshold(self) -> float Θ
#    realign(self, R, delta) -> new V
#    collapse(self, R) -> (new V, new E)   <-- NOTE: R is passed by Cognize
def collapse_fn(self, R):
    return collapse_slow(self.V, self.E)

def realign_fn(self, R, d):
    return realign_gravity(self.V, R, self.E)

def threshold_fn(self):
    return adaptive_threshold(self.E)

# 3) Create agent
newton = EpistemicState(V0=0.20, threshold=0.40)
newton.inject_policy(threshold=threshold_fn,
                     realign=realign_fn,
                     collapse=collapse_fn)

# 4) Evidence sequence
observations = [
    ("apple_fall_small",   0.25),
    ("apple_fall_repeat",  0.30),
    ("cannonball_exp",     0.45),
    ("moon_motion_hint",   0.55),
    ("tide_correlation",   0.85),  # likely rupture
    ("planet_orbit_calc",  0.92)   # consolidation
]

# 5) Run & trace
trace = []
for t, (evt, R) in enumerate(observations, start=1):
    V_before = float(newton.V)
    newton.receive(R, source=evt)
    ev = newton.last() or {}
    row = {
        "t": t,
        "event": evt,
        "R": float(ev.get("R", R)),
        "V_before": V_before,
        "V_after": float(ev.get("V", newton.V)),
        "Δ": float(ev.get("∆", abs(R - V_before))),
        "Θ": float(ev.get("Θ", getattr(newton, "Θ", 0.0))),
        "E": float(getattr(newton, "E", 0.0)),
        "rupture": bool(ev.get("rupture", ev.get("ruptured", False))),
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    trace.append(row)

# 6) Readable log
print("\n--- Mini-Newton Brain (scalar) ---")
print("t  event                  R     V_before  V_after   Δ      Θ      E     rupture")
for r in trace:
    print(f"{r['t']:>1}  {r['event']:<20}  {r['R']:.2f}  {r['V_before']:.2f}   "
          f"{r['V_after']:.2f}  {r['Δ']:.2f}  {r['Θ']:.2f}  {r['E']:.2f}   "
          f"{'YES' if r['rupture'] else 'no'}")

# 7) Plot: R vs V with rupture markers + E
times = [r["t"] for r in trace]
R_seq = [r["R"] for r in trace]
V_seq = [r["V_after"] for r in trace]
E_seq = [r["E"] for r in trace]
rupts = [r["t"] for r in trace if r["rupture"]]

fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)

ax[0].plot(times, R_seq, marker="s", label="R (reality)")
ax[0].plot(times, V_seq, marker="o", label="V (belief)")
for rt in rupts:
    ax[0].axvline(rt, linestyle="--", alpha=0.35)
ax[0].set_ylabel("value")
ax[0].set_title("Mini-Newton Brain — Reality vs Belief (ruptures dashed)")
ax[0].legend()

ax[1].plot(times, E_seq, marker=".", label="E (misalignment)")
ax[1].set_ylabel("E")
ax[1].set_xlabel("time / event index")
ax[1].legend()

plt.xticks(times, [r["event"] for r in trace], rotation=20, ha="right")
plt.tight_layout()
plt.show()
