"""Publication-quality figures for Phase 4 biological validation.

Generates multi-panel figures suitable for Nature Methods:
  Fig A – PSD overlay (real vs predicted) for each dataset
  Fig B – Band-R² heatmap across datasets × frequency bands
  Fig C – PAC comparison (real vs predicted MI)
  Fig D – CCA / DCCA / UNet R² bar chart
  Fig E – Decoding accuracy: real vs predicted vs cross-domain
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

from phase_four.config import DATASETS, FREQUENCY_BANDS, Phase4Config


# ── Style ─────────────────────────────────────────────────────────────────
NATURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Colour-blind safe palette (Wong 2011)
COLOURS = {
    "real": "#0072B2",
    "pred": "#D55E00",
    "cca": "#009E73",
    "dcca": "#CC79A7",
    "unet": "#E69F00",
    "chance": "#999999",
}


def _apply_style():
    if MPL_AVAILABLE:
        plt.rcParams.update(NATURE_STYLE)


# ═══════════════════════════════════════════════════════════════════════════
# A.  PSD overlay
# ═══════════════════════════════════════════════════════════════════════════

def plot_psd_overlay(
    spectral_results: Dict[str, Dict],
    dataset_specs: Dict[str, Any],
    out_path: Path,
):
    """PSD curves: real (blue) vs predicted (orange) for each dataset."""
    if not MPL_AVAILABLE:
        return
    _apply_style()

    datasets = list(spectral_results.keys())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.8), squeeze=False)

    for i, ds_key in enumerate(datasets):
        ax = axes[0, i]
        data = spectral_results[ds_key]
        freqs = np.array(data["freqs"])
        psd_real = np.array(data["psd_real"])
        psd_pred = np.array(data["psd_pred"])

        ax.semilogy(freqs, psd_real, color=COLOURS["real"], label="Real")
        ax.semilogy(freqs, psd_pred, color=COLOURS["pred"], label="Predicted", linestyle="--")
        ax.set_xlabel("Frequency (Hz)")
        if i == 0:
            ax.set_ylabel("PSD (V²/Hz)")
        ds_spec = dataset_specs.get(ds_key)
        title = ds_spec.display_name if ds_spec else ds_key
        ax.set_title(f"{title}\nr={data['psd_corr']:.3f}")
        ax.legend(frameon=False)
        ax.set_xlim(0, min(freqs[-1], 200))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# B.  Band-R² heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_band_r2_heatmap(
    spectral_results: Dict[str, Dict],
    dataset_specs: Dict[str, Any],
    out_path: Path,
):
    """Heatmap: datasets (rows) × frequency bands (cols), colour = R²."""
    if not MPL_AVAILABLE:
        return
    _apply_style()

    datasets = list(spectral_results.keys())
    bands = list(FREQUENCY_BANDS.keys())
    matrix = np.zeros((len(datasets), len(bands)))

    for i, ds in enumerate(datasets):
        for j, b in enumerate(bands):
            matrix[i, j] = spectral_results[ds]["band_r2"].get(b, 0.0)

    fig, ax = plt.subplots(figsize=(4.5, 0.5 * len(datasets) + 1.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=1.0)

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.replace("_", "\n") for b in bands])
    ylabels = [dataset_specs[d].display_name if d in dataset_specs else d for d in datasets]
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(ylabels)

    # Annotate cells
    for i in range(len(datasets)):
        for j in range(len(bands)):
            val = matrix[i, j]
            colour = "white" if val < 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=colour)

    fig.colorbar(im, ax=ax, label="R²", shrink=0.8)
    ax.set_title("Per-Band R²: Real vs Predicted Target")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# C.  PAC comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_pac_comparison(
    spectral_results: Dict[str, Dict],
    dataset_specs: Dict[str, Any],
    out_path: Path,
):
    """Paired bar chart: real MI vs predicted MI per dataset."""
    if not MPL_AVAILABLE:
        return
    _apply_style()

    datasets = list(spectral_results.keys())
    mi_real = [spectral_results[d]["pac_real"]["mi"] for d in datasets]
    mi_pred = [spectral_results[d]["pac_pred"]["mi"] for d in datasets]

    x = np.arange(len(datasets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.bar(x - w / 2, mi_real, w, color=COLOURS["real"], label="Real")
    ax.bar(x + w / 2, mi_pred, w, color=COLOURS["pred"], label="Predicted")

    labels = [dataset_specs[d].display_name if d in dataset_specs else d for d in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Modulation Index")
    ax.set_title("Phase-Amplitude Coupling Preservation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# D.  CCA / DCCA / UNet R² comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_cca_comparison(
    cca_results: Dict[str, Dict],
    dataset_specs: Dict[str, Any],
    out_path: Path,
):
    """Grouped bar: CCA R², DCCA R², UNet R² per dataset."""
    if not MPL_AVAILABLE:
        return
    _apply_style()

    datasets = list(cca_results.keys())
    r2_cca = [cca_results[d]["linear_cca"]["r2_cca"] for d in datasets]
    r2_unet = [cca_results[d]["linear_cca"]["r2_unet"] for d in datasets]
    r2_dcca = [cca_results[d]["deep_cca"].get("r2_dcca", 0.0) for d in datasets]

    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.bar(x - w, r2_cca, w, color=COLOURS["cca"], label="CCA")
    ax.bar(x, r2_dcca, w, color=COLOURS["dcca"], label="DCCA")
    ax.bar(x + w, r2_unet, w, color=COLOURS["unet"], label="NeuroGate")

    labels = [dataset_specs[d].display_name if d in dataset_specs else d for d in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Reconstruction Quality: CCA vs DCCA vs NeuroGate")
    ax.legend(frameon=False)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# E.  Decoding accuracy
# ═══════════════════════════════════════════════════════════════════════════

def plot_decoding_accuracy(
    decoding_results: Dict[str, Dict],
    dataset_specs: Dict[str, Any],
    out_path: Path,
):
    """Grouped bar: decoding accuracy from real / predicted / cross."""
    if not MPL_AVAILABLE:
        return
    _apply_style()

    # Only datasets with decoding results (skip those with < 2 classes)
    datasets = [d for d in decoding_results if not decoding_results[d].get("skipped")]
    if not datasets:
        print("  No datasets with decoding results to plot.")
        return

    # Use best classifier per dataset (highest real accuracy)
    acc_real, acc_pred, acc_cross, chance_levels = [], [], [], []
    for d in datasets:
        res = decoding_results[d]
        chance_levels.append(res["chance_level"])
        # Find best classifier
        best_clf = max(
            res["classifiers"],
            key=lambda c: res["classifiers"][c]["acc_real_cv"],
        )
        clf_res = res["classifiers"][best_clf]
        acc_real.append(clf_res["acc_real_cv"])
        acc_pred.append(clf_res["acc_pred_cv"])
        acc_cross.append(clf_res["acc_real_to_pred"])

    x = np.arange(len(datasets))
    w = 0.22
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.bar(x - w, acc_real, w, color=COLOURS["real"], label="Real targets")
    ax.bar(x, acc_pred, w, color=COLOURS["pred"], label="Predicted targets")
    ax.bar(x + w, acc_cross, w, color=COLOURS["unet"], label="Cross-domain")

    # Chance lines
    for i, ch in enumerate(chance_levels):
        ax.plot([i - 0.4, i + 0.4], [ch, ch], color=COLOURS["chance"],
                linestyle=":", linewidth=0.8)

    labels = [dataset_specs[d].display_name if d in dataset_specs else d for d in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Single-Trial Decoding from Translated Signals")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Master figure generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_figures(cfg: Phase4Config) -> None:
    """Load all saved results and generate publication figures."""
    if not MPL_AVAILABLE:
        print("matplotlib not available – skipping figures")
        return

    out = cfg.output_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    # Collect per-dataset results
    spectral_all, cca_all, decoding_all = {}, {}, {}
    # Also store PSD arrays for the overlay plot
    spectral_with_arrays = {}

    for ds_key in cfg.datasets:
        sd = cfg.get_synth_dir(ds_key)
        sp = sd / "spectral_results.json"
        if sp.exists():
            with open(sp) as f:
                spectral_all[ds_key] = json.load(f)
            # Load full PSD data from npz if saved, else re-compute
            spectral_with_arrays[ds_key] = spectral_all[ds_key]

        cp = sd / "cca_results.json"
        if cp.exists():
            with open(cp) as f:
                cca_all[ds_key] = json.load(f)

        dp = sd / "decoding_results.json"
        if dp.exists():
            with open(dp) as f:
                decoding_all[ds_key] = json.load(f)

    specs = DATASETS

    if spectral_all:
        # For PSD overlay we need the actual arrays – re-compute quickly
        psd_data = {}
        for ds_key in spectral_all:
            sd = cfg.get_synth_dir(ds_key)
            target = np.load(sd / "target_test.npy")
            pred = np.load(sd / "predicted_test.npy")
            fs = specs[ds_key].sampling_rate if ds_key in specs else 1000
            from phase_four.validation.spectral import compute_psd
            freqs, psd_real = compute_psd(target, fs, cfg.nperseg)
            _, psd_pred = compute_psd(pred, fs, cfg.nperseg)
            psd_data[ds_key] = {
                "freqs": freqs.tolist(),
                "psd_real": psd_real.tolist(),
                "psd_pred": psd_pred.tolist(),
                "psd_corr": spectral_all[ds_key]["psd_corr"],
            }

        plot_psd_overlay(psd_data, specs, out / "fig_psd_overlay.pdf")
        plot_band_r2_heatmap(spectral_all, specs, out / "fig_band_r2_heatmap.pdf")
        plot_pac_comparison(spectral_all, specs, out / "fig_pac_comparison.pdf")

    if cca_all:
        plot_cca_comparison(cca_all, specs, out / "fig_cca_comparison.pdf")

    if decoding_all:
        plot_decoding_accuracy(decoding_all, specs, out / "fig_decoding_accuracy.pdf")

    print(f"\nAll figures saved to {out}/")
