"""CCA / DCCA baseline comparison.

Fits linear CCA and Deep CCA on source→target pairs, then compares the
canonical correlations and reconstruction R² against the UNet predictions.

This answers the reviewer question: "How much of the translation can a
simple linear (or shallow nonlinear) subspace method capture?"
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cross_decomposition import CCA

# ── optional deep CCA (torch) ────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Linear CCA
# ═══════════════════════════════════════════════════════════════════════════

def _flatten_trials(x: np.ndarray) -> np.ndarray:
    """[N, C, T] → [N*T, C]  (samples × features)."""
    N, C, T = x.shape
    return x.transpose(0, 2, 1).reshape(-1, C)


def linear_cca(
    source: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    n_components: int = 10,
) -> Dict[str, Any]:
    """Fit CCA on source-target, project predicted, measure reconstruction.

    Strategy:
        1. Fit CCA(source, target) on flattened test data (this is the
           *best-case* linear baseline — it sees the test targets).
        2. Project source through CCA → reconstruct target → measure R².
        3. Compare CCA-reconstructed target vs UNet-predicted target.

    We intentionally fit CCA *on the test set* to give the linear baseline
    its best possible shot.  The UNet never saw the test set during training,
    so any gap is a genuine advantage of the learned model.
    """
    n_components = min(n_components, source.shape[1], target.shape[1])

    src_flat = _flatten_trials(source)   # [N*T, C_src]
    tgt_flat = _flatten_trials(target)   # [N*T, C_tgt]
    pred_flat = _flatten_trials(pred)    # [N*T, C_tgt]

    # Sub-sample if very large (CCA scales as O(n*d²))
    max_samples = 200_000
    if src_flat.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(src_flat.shape[0], max_samples, replace=False)
        src_fit = src_flat[idx]
        tgt_fit = tgt_flat[idx]
    else:
        src_fit = src_flat
        tgt_fit = tgt_flat

    cca = CCA(n_components=n_components, max_iter=1000)
    cca.fit(src_fit, tgt_fit)

    # Canonical correlations on full data
    src_proj, tgt_proj = cca.transform(src_flat, tgt_flat)
    canon_corrs = []
    for i in range(n_components):
        r = np.corrcoef(src_proj[:, i], tgt_proj[:, i])[0, 1]
        canon_corrs.append(float(r))

    # Reconstruct target from source via CCA
    # CCA gives us: src_proj = src_flat @ x_weights_
    #               tgt_proj = tgt_flat @ y_weights_
    # To reconstruct: tgt_hat = src_proj @ pinv(y_weights_)
    y_weights = cca.y_weights_                         # [C_tgt, n_comp]
    tgt_hat_flat = src_proj @ np.linalg.pinv(y_weights.T)  # [N*T, C_tgt]

    # R² of CCA reconstruction
    ss_res_cca = np.sum((tgt_flat - tgt_hat_flat) ** 2)
    ss_tot = np.sum((tgt_flat - tgt_flat.mean(axis=0)) ** 2)
    r2_cca = 1.0 - ss_res_cca / (ss_tot + 1e-20)

    # R² of UNet prediction
    ss_res_unet = np.sum((tgt_flat - pred_flat) ** 2)
    r2_unet = 1.0 - ss_res_unet / (ss_tot + 1e-20)

    # Per-channel correlations and R²
    per_channel = []
    for c in range(tgt_flat.shape[1]):
        t_ch = tgt_flat[:, c]
        cca_ch = tgt_hat_flat[:, c]
        unet_ch = pred_flat[:, c]

        corr_cca = float(np.corrcoef(t_ch, cca_ch)[0, 1])
        corr_unet = float(np.corrcoef(t_ch, unet_ch)[0, 1])

        ss_tot_ch = np.sum((t_ch - t_ch.mean()) ** 2) + 1e-20
        r2_cca_ch = float(1.0 - np.sum((t_ch - cca_ch) ** 2) / ss_tot_ch)
        r2_unet_ch = float(1.0 - np.sum((t_ch - unet_ch) ** 2) / ss_tot_ch)

        per_channel.append({
            "channel": c,
            "corr_cca": corr_cca,
            "corr_unet": corr_unet,
            "r2_cca": r2_cca_ch,
            "r2_unet": r2_unet_ch,
            "r2_gain": r2_unet_ch - r2_cca_ch,
        })

    corr_cca_per_ch = [ch["corr_cca"] for ch in per_channel]
    corr_unet_per_ch = [ch["corr_unet"] for ch in per_channel]

    return {
        "n_components": n_components,
        "canonical_correlations": canon_corrs,
        "r2_cca": float(r2_cca),
        "r2_unet": float(r2_unet),
        "r2_gain_over_cca": float(r2_unet - r2_cca),
        "mean_corr_cca": float(np.mean(corr_cca_per_ch)),
        "mean_corr_unet": float(np.mean(corr_unet_per_ch)),
        "corr_cca_per_channel": corr_cca_per_ch,
        "corr_unet_per_channel": corr_unet_per_ch,
        "per_channel": per_channel,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Deep CCA  (two-view MLP with CCA objective)
# ═══════════════════════════════════════════════════════════════════════════

class _DCCAEncoder(nn.Module):
    """Simple 3-layer MLP for one view."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _cca_loss(H1: "torch.Tensor", H2: "torch.Tensor", eps: float = 1e-4):
    """Differentiable CCA loss (negative sum of top-k correlations).

    Andrew et al. (2013) "Deep Canonical Correlation Analysis".
    """
    N = H1.size(0)
    d = H1.size(1)

    # Centre
    H1 = H1 - H1.mean(dim=0, keepdim=True)
    H2 = H2 - H2.mean(dim=0, keepdim=True)

    S11 = (H1.T @ H1) / (N - 1) + eps * torch.eye(d, device=H1.device)
    S22 = (H2.T @ H2) / (N - 1) + eps * torch.eye(d, device=H1.device)
    S12 = (H1.T @ H2) / (N - 1)

    # Whitening
    D1, V1 = torch.linalg.eigh(S11)
    D2, V2 = torch.linalg.eigh(S22)
    D1 = D1.clamp(min=eps)
    D2 = D2.clamp(min=eps)

    S11_inv_half = V1 @ torch.diag(D1 ** -0.5) @ V1.T
    S22_inv_half = V2 @ torch.diag(D2 ** -0.5) @ V2.T

    T = S11_inv_half @ S12 @ S22_inv_half
    # Sum of singular values = sum of canonical correlations
    U, S, Vt = torch.linalg.svd(T)
    return -S.sum()


def deep_cca(
    source: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    n_components: int = 10,
    hidden_dim: int = 256,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 2048,
) -> Dict[str, Any]:
    """Train Deep CCA on source-target, compare reconstruction to UNet.

    Returns metrics dict analogous to linear_cca().
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available for DCCA"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_flat = _flatten_trials(source).astype(np.float32)
    tgt_flat = _flatten_trials(target).astype(np.float32)
    pred_flat = _flatten_trials(pred).astype(np.float32)

    # Sub-sample for training
    max_samples = 200_000
    rng = np.random.default_rng(42)
    if src_flat.shape[0] > max_samples:
        idx = rng.choice(src_flat.shape[0], max_samples, replace=False)
        src_train = src_flat[idx]
        tgt_train = tgt_flat[idx]
    else:
        src_train = src_flat
        tgt_train = tgt_flat

    n_components = min(n_components, source.shape[1], target.shape[1])

    enc_src = _DCCAEncoder(src_train.shape[1], hidden_dim, n_components).to(device)
    enc_tgt = _DCCAEncoder(tgt_train.shape[1], hidden_dim, n_components).to(device)
    optimizer = torch.optim.Adam(
        list(enc_src.parameters()) + list(enc_tgt.parameters()), lr=lr
    )

    ds = TensorDataset(
        torch.from_numpy(src_train),
        torch.from_numpy(tgt_train),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Train
    enc_src.train()
    enc_tgt.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for s_batch, t_batch in loader:
            s_batch = s_batch.to(device)
            t_batch = t_batch.to(device)
            h1 = enc_src(s_batch)
            h2 = enc_tgt(t_batch)
            loss = _cca_loss(h1, h2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # Evaluate – project full data and reconstruct target
    enc_src.eval()
    enc_tgt.eval()
    with torch.no_grad():
        # Process in batches to avoid OOM
        h1_all, h2_all = [], []
        for i in range(0, src_flat.shape[0], batch_size):
            s = torch.from_numpy(src_flat[i:i+batch_size]).to(device)
            t = torch.from_numpy(tgt_flat[i:i+batch_size]).to(device)
            h1_all.append(enc_src(s).cpu().numpy())
            h2_all.append(enc_tgt(t).cpu().numpy())
        h1_np = np.concatenate(h1_all)
        h2_np = np.concatenate(h2_all)

    # Canonical correlations in learned space
    canon_corrs = []
    for i in range(n_components):
        r = np.corrcoef(h1_np[:, i], h2_np[:, i])[0, 1]
        canon_corrs.append(float(r))

    # Reconstruct target from source embeddings via linear regression
    # h1 → tgt_flat  (linear decoder)
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=1.0)
    reg.fit(h1_np, tgt_flat)
    tgt_hat = reg.predict(h1_np)

    ss_res = np.sum((tgt_flat - tgt_hat) ** 2)
    ss_tot = np.sum((tgt_flat - tgt_flat.mean(axis=0)) ** 2)
    r2_dcca = 1.0 - ss_res / (ss_tot + 1e-20)

    ss_res_unet = np.sum((tgt_flat - pred_flat) ** 2)
    r2_unet = 1.0 - ss_res_unet / (ss_tot + 1e-20)

    return {
        "n_components": n_components,
        "canonical_correlations": canon_corrs,
        "r2_dcca": float(r2_dcca),
        "r2_unet": float(r2_unet),
        "r2_gain_over_dcca": float(r2_unet - r2_dcca),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_cca_validation(
    synth_dir: Path,
    n_components: int = 10,
    dcca_hidden_dim: int = 256,
    dcca_epochs: int = 100,
    dcca_lr: float = 1e-3,
) -> Dict[str, Any]:
    """Run CCA and DCCA comparison on saved synthetic data.

    Args:
        synth_dir: path containing source_test.npy, target_test.npy,
                   predicted_test.npy
    """
    source = np.load(synth_dir / "source_test.npy")
    target = np.load(synth_dir / "target_test.npy")
    pred = np.load(synth_dir / "predicted_test.npy")

    print(f"  CCA validation: source {source.shape}, target {target.shape}")

    # Linear CCA
    print("    Fitting linear CCA...")
    cca_results = linear_cca(source, target, pred, n_components)
    print(f"    CCA  R² : {cca_results['r2_cca']:.4f}")
    print(f"    UNet R² : {cca_results['r2_unet']:.4f}")
    print(f"    Gain    : {cca_results['r2_gain_over_cca']:+.4f}")
    print(f"    Top-3 canonical corrs: "
          f"{cca_results['canonical_correlations'][:3]}")

    # Deep CCA
    print("    Training Deep CCA...")
    dcca_results = deep_cca(
        source, target, pred,
        n_components=n_components,
        hidden_dim=dcca_hidden_dim,
        epochs=dcca_epochs,
        lr=dcca_lr,
    )
    if "error" not in dcca_results:
        print(f"    DCCA R² : {dcca_results['r2_dcca']:.4f}")
        print(f"    Gain    : {dcca_results['r2_gain_over_dcca']:+.4f}")

    results = {
        "linear_cca": cca_results,
        "deep_cca": dcca_results,
    }

    out_path = synth_dir / "cca_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    → saved {out_path}")

    # Per-channel CSV
    _save_cca_channel_csv(cca_results, synth_dir / "cca_per_channel.csv")

    return results


def _save_cca_channel_csv(cca_results: Dict, out_path: Path):
    """Write per-channel CCA vs UNet metrics as CSV."""
    import csv
    per_ch = cca_results.get("per_channel", [])
    if not per_ch:
        return
    header = ["channel", "corr_cca", "corr_unet", "r2_cca", "r2_unet", "r2_gain"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ch in per_ch:
            writer.writerow([
                f"ch_{ch['channel']}",
                f"{ch['corr_cca']:.6f}",
                f"{ch['corr_unet']:.6f}",
                f"{ch['r2_cca']:.6f}",
                f"{ch['r2_unet']:.6f}",
                f"{ch['r2_gain']:.6f}",
            ])
    print(f"    → saved {out_path}")
