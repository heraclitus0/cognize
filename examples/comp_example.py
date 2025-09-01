# 1) Setup
try:
    import cognize  # if already installed
except Exception:
    !pip -q install cognize numpy pandas matplotlib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

from cognize import (
    EpistemicState, Perception,
    PolicyManager, PolicyMemory, ShadowRunner, SAFE_SPECS
)
from cognize.policies import threshold_adaptive, realign_tanh, collapse_soft_decay

# Repro + output dir
rng = np.random.default_rng(42)
out_dir = Path("./cognize_totality"); out_dir.mkdir(parents=True, exist_ok=True)

# 2) Generate a reality stream with regime shifts (no models involved)
T = 600
x = np.linspace(0, 60, T)
R = np.sin(x) + 0.08 * rng.standard_normal(T)
R[200:360] += 1.8   # regime up-shift
R[360:]     -= 1.1  # regime down-shift

# 3) Baseline governor (fixed policy)
state = EpistemicState(V0=0.0, threshold=0.35, realign_strength=0.30, rng_seed=123)
state.inject_policy(threshold=threshold_adaptive, realign=realign_tanh, collapse=collapse_soft_decay)
for r in R: state.receive(r)
base_csv = str(out_dir / "baseline.csv")
state.export_csv(base_csv)
print("Baseline:", state.summary())

# ---- helpers for reading/plotting/export stats
def read_cols(csv_path):
    df = pd.read_csv(csv_path)
    def pick(keys, default=None):
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in keys): return c
        return default
    cols = dict(
        R  = pick(["reality"," r","input"]) or df.columns[0],
        V  = pick(["belief"," v","state"]),
        E  = pick(["misalignment"," e","memory"]),
        Ru = pick(["rupture","collapse","trigger"]),
    )
    return df, cols

def plot_trace(df, cols, title=""):
    rows = 3
    fig, ax = plt.subplots(rows, 1, figsize=(12, 3.2*rows), sharex=True)
    ax[0].plot(df[cols["R"]].values, label="R (reality)")
    if cols["V"] and cols["V"] in df: ax[0].plot(df[cols["V"]].values, label="V (belief)")
    ax[0].legend(); ax[0].set_title(f"Reality vs Belief {title}")

    ax[1].plot(df[cols["Ru"]].values if cols["Ru"] in df else np.zeros(len(df)), label="rupture")
    ax[1].set_ylim(-0.1, 1.2); ax[1].legend(); ax[1].set_title("Rupture events")

    if cols["E"] and cols["E"] in df:
        ax[2].plot(df[cols["E"]].values, label="E (misalignment)")
        ax[2].legend(); ax[2].set_title("Misalignment memory E")
    plt.tight_layout(); plt.show()

def stats(csv_path):
    d, c = read_cols(csv_path)
    rupt = int(d[c["Ru"]].fillna(0).round().sum()) if c["Ru"] in d else 0
    emax = float(d[c["E"]].max()) if c["E"] in d else float("nan")
    return d, c, rupt, emax

# Plot baseline & print stats
dfb, cols_b = read_cols(base_csv)
plot_trace(dfb, cols_b, "(baseline)")
_, _, rupt_b, emax_b = stats(base_csv)
print(f"Baseline -> ruptures={rupt_b}, E_max={emax_b:.4f}")

# 4) Θ sweep — governance sensitivity
def run_theta(theta):
    s = EpistemicState(V0=0.0, threshold=theta, realign_strength=0.30, rng_seed=7)
    s.inject_policy(threshold=threshold_adaptive, realign=realign_tanh, collapse=collapse_soft_decay)
    for r in R: s.receive(r)
    tmp = str(out_dir / "theta.csv"); s.export_csv(tmp)
    _, _, rupt, emax = stats(tmp)
    return dict(Theta=theta, Ruptures=rupt, E_max=emax)

thetas = [0.20, 0.30, 0.35, 0.45, 0.60]
theta_df = pd.DataFrame([run_theta(t) for t in thetas])
print("\nΘ sweep:")
print(theta_df.to_string(index=False))

# 5) Meta-policy evolution (SAFE_SPECS + shadow evaluation)
state2 = EpistemicState(V0=0.0, threshold=0.35, realign_strength=0.30, rng_seed=99)
state2.inject_policy(threshold=threshold_adaptive, realign=realign_tanh, collapse=collapse_soft_decay)
state2.policy_manager = PolicyManager(
    base_specs=SAFE_SPECS,
    memory=PolicyMemory(),
    shadow=ShadowRunner(),
    epsilon=0.15,         # exploration rate
    promote_margin=1.03,  # must beat baseline by 3%
    cooldown_steps=30
)
for r in R: state2.receive(r)
evo_csv = str(out_dir / "evolved.csv")
state2.export_csv(evo_csv)

dfe, cols_e = read_cols(evo_csv)
plot_trace(dfe, cols_e, "(evolved meta-policy)")
_, _, rupt_e, emax_e = stats(evo_csv)
print(f"\nEvolved -> ruptures={rupt_e}, E_max={emax_e:.4f}")
print(f"Δ vs baseline -> ruptures={rupt_e-rupt_b}, E_max={emax_e-emax_b:+.4f}")

# Show last policy evolution events if available
try:
    tail = state2.event_log_summary()[-8:]
    print("\nPolicy evolution tail:")
    for e in tail: print(e)
except Exception:
    pass

# 6) Optional — Perception demo (dict -> fused vector). Toggle to True to run.
DO_PERCEPTION = False
if DO_PERCEPTION:
    def toy_text_encoder(s: str) -> np.ndarray:
        return np.array([len(s), s.count(" "), s.count("a"), 1.0], dtype=float)
    P = Perception(text_encoder=toy_text_encoder)
    S = EpistemicState(V0=np.zeros(4), perception=P)
    for ex in [{"text":"hello world"}, {"text":"a a a a"}, {"text":"short"}]:
        S.receive(ex)
    print("\nPerception last:", S.explain_last())

print(f"\nCSV outputs in: {out_dir.resolve()}")
