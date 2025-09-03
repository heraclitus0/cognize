# ======================================================================
# Cognize Total Demo — Baseline → Θ-sweep → Meta-policy Evolution
# ======================================================================
"""
Cognize Total Demo
==================

This example shows how to:

• Track belief vs. reality and detect ruptures  
• Explore how different thresholds (Θ) affect stability  
• Let policies evolve through shadow evaluation and safe promotion  

Outputs:
  • CSV traces for baseline, evolved, and Θ-sweep  
  • Plots of reality vs. belief, rupture events, and misalignment memory  
  • Comparison charts for baseline vs. evolved  

Run from CLI:
    python -m cognize.examples.total_demo --steps 600 --lam 0.5 --out ./cognize_totality

Run inside notebook/Colab:
    from cognize.examples import total_demo
    total_demo.main()
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cognize import (
    EpistemicState, PolicyManager, PolicyMemory, ShadowRunner, SAFE_SPECS
)
from cognize.policies import threshold_adaptive, realign_tanh, collapse_soft_decay


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_cols(csv_path):
    df = pd.read_csv(csv_path)
    lc = {c.lower(): c for c in df.columns}

    def find(*keys, default=None):
        for k in keys:
            for lc_name, orig in lc.items():
                if k in lc_name:
                    return orig
        return default

    return df, {
        "R": find("reality", "input", "r", "x"),
        "V": find("belief", "v", "pred", "estimate"),
        "E": find("misalignment", "memory", "e"),
        "Ru": find("rupture", "collapse", "trigger", "reset"),
        "Delta": find("Δ", "delta", "drift"),
    }


def compute_metrics(df: pd.DataFrame, cols: Dict[str, str], lam: float = 0.5) -> Dict[str, float]:
    """Compute drift, rupture frequency, memory peak, and composite score J."""
    # Drift Δ
    if cols["Delta"] and cols["Delta"] in df:
        delta = pd.to_numeric(df[cols["Delta"]], errors="coerce").fillna(0.0).values
    elif cols["R"] and cols["V"] and cols["R"] in df and cols["V"] in df:
        r = pd.to_numeric(df[cols["R"]], errors="coerce").fillna(0.0).values
        v = pd.to_numeric(df[cols["V"]], errors="coerce").fillna(0.0).values
        delta = r - v
    else:
        delta = np.zeros(len(df))
    abs_delta = np.abs(delta)
    mean_abs_delta = float(np.mean(abs_delta))

    # Ruptures
    if cols["Ru"] and cols["Ru"] in df:
        ru = pd.to_numeric(df[cols["Ru"]], errors="coerce").fillna(0.0)
        ruptures = (ru > 0.5).astype(int)
    else:
        ruptures = pd.Series([0] * len(df), dtype=int)
    ruptures_total = int(ruptures.sum())
    ruptures_per_100 = 100.0 * ruptures_total / max(1, len(df))

    # Misalignment memory
    if cols["E"] and cols["E"] in df:
        e_max = float(pd.to_numeric(df[cols["E"]], errors="coerce").max())
    else:
        e_max = float("nan")

    # Composite score
    J = mean_abs_delta + lam * ruptures_per_100
    return dict(J=J, mean_abs_delta=mean_abs_delta,
                ruptures_total=ruptures_total, ruptures_per_100=ruptures_per_100,
                E_max=e_max, steps=len(df))


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_trace(df: pd.DataFrame, cols: Dict[str, str], title: str = "", out_path: Path | None = None) -> None:
    """Plot reality vs. belief, ruptures, and misalignment memory."""
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    r = pd.to_numeric(df[cols["R"]], errors="coerce").fillna(0.0).values
    ax[0].plot(r, label="R (reality)")
    if cols["V"] and cols["V"] in df:
        v = pd.to_numeric(df[cols["V"]], errors="coerce").fillna(0.0).values
        ax[0].plot(v, label="V (belief)")
    ax[0].legend(); ax[0].set_title(f"Reality vs Belief {title}")

    if cols["Ru"] and cols["Ru"] in df:
        ru = pd.to_numeric(df[cols["Ru"]], errors="coerce").fillna(0.0).values
        ax[1].plot(ru, label="rupture")
    ax[1].set_ylim(-0.1, 1.2); ax[1].legend(); ax[1].set_title("Rupture events")

    if cols["E"] and cols["E"] in df:
        e = pd.to_numeric(df[cols["E"]], errors="coerce").fillna(0.0).values
        ax[2].plot(e, label="E (misalignment)")
        ax[2].legend(); ax[2].set_title("Misalignment memory")

    plt.tight_layout()
    if out_path: plt.savefig(out_path, dpi=160)
    plt.show()


def bar_compare(mb: Dict[str, float], me: Dict[str, float], out_path: Path | None = None) -> None:
    """Side-by-side bar chart for baseline vs evolved metrics."""
    labels = ["mean|Δ|", "ruptures/100", "E_max", "J"]
    b = [mb["mean_abs_delta"], mb["ruptures_per_100"], mb["E_max"], mb["J"]]
    e = [me["mean_abs_delta"], me["ruptures_per_100"], me["E_max"], me["J"]]

    x = np.arange(len(labels)); w = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - w/2, b, width=w, label="baseline")
    plt.bar(x + w/2, e, width=w, label="evolved")
    plt.xticks(x, labels); plt.legend(); plt.title("Baseline vs Evolved")
    plt.tight_layout()
    if out_path: plt.savefig(out_path, dpi=160)
    plt.show()


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def make_reality(T: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic signal with mid up-shift and late down-shift."""
    x = np.linspace(0, 60, T)
    R = np.sin(x) + 0.08 * rng.standard_normal(T)
    if T >= 360:
        R[int(T*0.33):int(T*0.60)] += 1.8
        R[int(T*0.60):] -= 1.1
    return R


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lam", type=float, default=0.5, help="rupture penalty in J")
    p.add_argument("--out", type=str, default="./cognize_totality")
    p.add_argument("--seed", type=int, default=42)

    # Works in both CLI and Jupyter/Colab
    args, _ = p.parse_known_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate stream
    R = make_reality(args.steps, rng)

    # 2) Baseline
    state = EpistemicState(V0=0.0, threshold=0.35, realign_strength=0.30, rng_seed=123)
    state.inject_policy(threshold=threshold_adaptive, realign=realign_tanh, collapse=collapse_soft_decay)
    for r in R: state.receive(r)
    base_csv = str(out_dir / "baseline.csv"); state.export_csv(base_csv)

    dfb, cols_b = read_cols(base_csv)
    metrics_b = compute_metrics(dfb, cols_b, lam=args.lam)
    plot_trace(dfb, cols_b, "(baseline)", out_path=out_dir/"trace_baseline.png")
    print("Baseline:", metrics_b)

    # 3) Θ-sweep
    def run_theta(theta):
        s = EpistemicState(V0=0.0, threshold=theta, realign_strength=0.30, rng_seed=7)
        s.inject_policy(threshold=threshold_adaptive, realign=realign_tanh, collapse=collapse_soft_decay)
        for r in R: s.receive(r)
        tmp = str(out_dir / f"theta_{theta:.2f}.csv"); s.export_csv(tmp)
        d, c = read_cols(tmp); m = compute_metrics(d, c, lam=args.lam); m["Theta"] = theta
        return m

    thetas = [0.20, 0.30, 0.35, 0.45, 0.60]
    theta_df = pd.DataFrame([run_theta(t) for t in thetas])[["Theta","J","mean_abs_delta","ruptures_per_100","E_max"]]
    theta_df.to_csv(out_dir/"theta_sweep.csv", index=False)
    print("\nΘ-sweep:\n", theta_df)

    # 4) Evolved (meta-policy)
    state2 = EpistemicState(V0=0.0, threshold=0.35, realign_strength=0.30, rng_seed=99)
    state2.inject_policy(threshold=threshold_adaptive, realign=realign_tanh, collapse=collapse_soft_decay)
    state2.policy_manager = PolicyManager(
        base_specs=SAFE_SPECS, memory=PolicyMemory(), shadow=ShadowRunner(),
        epsilon=0.15, promote_margin=1.03, cooldown_steps=30
    )
    for r in R: state2.receive(r)
    evo_csv = str(out_dir / "evolved.csv"); state2.export_csv(evo_csv)

    dfe, cols_e = read_cols(evo_csv)
    metrics_e = compute_metrics(dfe, cols_e, lam=args.lam)
    plot_trace(dfe, cols_e, "(evolved)", out_path=out_dir/"trace_evolved.png")
    bar_compare(metrics_b, metrics_e, out_path=out_dir/"compare.png")
    print("\nEvolved:", metrics_e)

    print(f"\nArtifacts saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
