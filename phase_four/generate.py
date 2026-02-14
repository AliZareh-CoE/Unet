"""Generate synthetic (predicted) target signals from trained models.

Loads a Phase-4 checkpoint, runs inference on the held-out 30 % test set,
and persists both real and predicted arrays to ``/data/synth/<dataset>/``.

Saved files per dataset
-----------------------
source_test.npy      – real source signals   [N, C_src, T]
target_test.npy      – real target signals    [N, C_tgt, T]
predicted_test.npy   – model predictions      [N, C_tgt, T]
labels_test.npy      – trial / window labels  [N]
metadata.json        – channel counts, sampling rate, regions, split info
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── project imports ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase_four.config import DATASETS, Phase4Config


def _build_model_from_config(config: dict, device: torch.device):
    """Reconstruct NeuroGate from saved checkpoint config dict."""
    from train import SAMPLING_RATE_HZ
    try:
        from models import NeuroGate
    except ImportError:
        from Unet.models import NeuroGate

    model = NeuroGate(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        base=config.get("base_channels", 128),
        n_odors=config.get("n_odors", 0),
        dropout=config.get("dropout", 0.0),
        use_attention=config.get("use_attention", True),
        attention_type=config.get("attention_type", "self_attn"),
        norm_type="batch",
        cond_mode=config.get("cond_mode", "film"),
        n_downsample=config.get("n_downsample", 2),
        conv_type=config.get("conv_type", "standard"),
        use_se=config.get("use_se", True),
        conv_kernel_size=config.get("conv_kernel_size", 7),
        dilations=config.get("conv_dilations", (1, 4, 16, 32)),
        use_session_stats=config.get("use_session_stats", False),
        session_emb_dim=config.get("session_emb_dim", 32),
        session_use_spectral=config.get("session_use_spectral", False),
        use_session_embedding=config.get("use_session_embedding", False),
        n_sessions=config.get("n_sessions", 0),
        use_adaptive_scaling=config.get("use_adaptive_scaling", False),
        adaptive_scaling_mode=config.get("adaptive_scaling_mode", "scalar"),
        adaptive_scaling_cross_channel=config.get("adaptive_scaling_cross_channel", False),
        activation=config.get("activation", "relu"),
        n_heads=config.get("n_heads", 4),
        skip_type=config.get("skip_type", "add"),
        sampling_rate=config.get("sampling_rate", SAMPLING_RATE_HZ),
    )
    model.to(device)
    model.eval()
    return model


def _load_model(checkpoint_path: Path, device: torch.device):
    """Load model + config from a Phase-4 checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = _build_model_from_config(config, device)
    model.load_state_dict(ckpt["model"])
    return model, config, ckpt


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run model on every batch, collect source / target / prediction arrays."""
    model.eval()
    sources, targets, preds, labels = [], [], [], []

    for batch in loader:
        src, tgt = batch[0].to(device), batch[1].to(device)
        label = batch[2] if len(batch) > 2 else torch.zeros(src.shape[0])

        pred = model(src)

        sources.append(src.cpu().numpy())
        targets.append(tgt.cpu().numpy())
        preds.append(pred.cpu().numpy())
        labels.append(label.numpy() if isinstance(label, torch.Tensor) else np.array(label))

    return {
        "source": np.concatenate(sources, axis=0),
        "target": np.concatenate(targets, axis=0),
        "predicted": np.concatenate(preds, axis=0),
        "labels": np.concatenate(labels, axis=0),
    }


def generate_and_save(
    dataset_key: str,
    cfg: Phase4Config,
    device: Optional[torch.device] = None,
) -> Path:
    """Full pipeline: load model → inference → save to /data/synth/<dataset>.

    Returns the output directory path.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = DATASETS[dataset_key]
    ckpt_path = cfg.get_checkpoint_path(dataset_key)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint for {dataset_key} at {ckpt_path}. "
            "Run training first (python -m phase_four.runner --train)."
        )

    print(f"\n{'='*60}")
    print(f"Generating synthetic signals: {ds.display_name}")
    print(f"  Checkpoint : {ckpt_path}")

    model, train_config, ckpt = _load_model(ckpt_path, device)

    # ── Rebuild test dataloader from saved split info ─────────────────────
    split_info = ckpt.get("split_info", {})
    test_idx = split_info.get("test_idx")
    if test_idx is None:
        raise RuntimeError(f"Checkpoint missing split_info.test_idx for {dataset_key}")

    loader = _make_test_loader(dataset_key, ds, train_config, test_idx, cfg)

    # ── Run inference ─────────────────────────────────────────────────────
    results = run_inference(model, loader, device)

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = cfg.get_synth_dir(dataset_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "source_test.npy", results["source"])
    np.save(out_dir / "target_test.npy", results["target"])
    np.save(out_dir / "predicted_test.npy", results["predicted"])
    np.save(out_dir / "labels_test.npy", results["labels"])

    metadata = {
        "dataset": dataset_key,
        "display_name": ds.display_name,
        "source_region": ds.source_region,
        "target_region": ds.target_region,
        "in_channels": results["source"].shape[1],
        "out_channels": results["target"].shape[1],
        "n_samples": int(results["source"].shape[0]),
        "time_steps": int(results["source"].shape[2]),
        "sampling_rate": ds.sampling_rate,
        "n_labels": int(np.unique(results["labels"]).shape[0]),
        "checkpoint": str(ckpt_path),
        "test_idx": [int(i) for i in test_idx],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved {results['source'].shape[0]} test samples → {out_dir}")
    print(f"  source  : {results['source'].shape}")
    print(f"  target  : {results['target'].shape}")
    print(f"  predicted: {results['predicted'].shape}")
    return out_dir


# ── Dataset-specific test-loader factories ────────────────────────────────

def _make_test_loader(
    dataset_key: str,
    ds,
    train_config: dict,
    test_idx,
    cfg: Phase4Config,
) -> DataLoader:
    """Create a test-only DataLoader for the given dataset."""
    import numpy as np

    test_idx = np.array(test_idx)

    if ds.train_name == "olfactory":
        return _olfactory_test_loader(test_idx, cfg)
    elif ds.train_name == "pfc":
        reverse = "--pfc-reverse" in ds.extra_train_args
        return _pfc_test_loader(test_idx, cfg, reverse=reverse)
    elif ds.train_name == "dandi":
        return _dandi_test_loader(test_idx, train_config, cfg)
    elif ds.train_name == "ecog":
        return _ecog_test_loader(test_idx, train_config, cfg)
    elif ds.train_name == "boran":
        return _boran_test_loader(test_idx, train_config, cfg)
    else:
        raise ValueError(f"Unknown train_name: {ds.train_name}")


def _olfactory_test_loader(test_idx, cfg):
    from data import load_all_sessions, PairedNeuralDataset
    ob, pcx, labels, _ = load_all_sessions()
    ds = PairedNeuralDataset(ob, pcx, labels, test_idx)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)


def _pfc_test_loader(test_idx, cfg, reverse=False):
    from data import prepare_pfc_data, PairedNeuralDataset
    data = prepare_pfc_data(split_by_session=True)
    if reverse:
        source, target = data["ca1"], data["pfc"]
    else:
        source, target = data["pfc"], data["ca1"]
    trial_types = data["trial_types"]
    ds = PairedNeuralDataset(source, target, trial_types, test_idx)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)


def _dandi_test_loader(test_idx, train_config, cfg):
    from data import prepare_dandi_data
    data = prepare_dandi_data(
        source_region=train_config.get("dandi_source_region", "amygdala"),
        target_region=train_config.get("dandi_target_region", "hippocampus"),
    )
    # DANDI uses sliding window datasets – test_idx refers to subject indices
    test_loader = data.get("test_loader")
    if test_loader is not None:
        return test_loader
    # Fallback: manually create from test split
    from torch.utils.data import Subset
    full_ds = data["dataset"]
    ds = Subset(full_ds, test_idx)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)


def _ecog_test_loader(test_idx, train_config, cfg):
    from data import prepare_ecog_data
    data = prepare_ecog_data(
        experiment=train_config.get("ecog_experiment", "motor_imagery"),
        source_region=train_config.get("ecog_source_region", "frontal"),
        target_region=train_config.get("ecog_target_region", "temporal"),
    )
    test_loader = data.get("test_loader")
    if test_loader is not None:
        return test_loader
    from torch.utils.data import Subset
    full_ds = data["dataset"]
    ds = Subset(full_ds, test_idx)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)


def _boran_test_loader(test_idx, train_config, cfg):
    from data import prepare_boran_data
    data = prepare_boran_data(
        source_region=train_config.get("boran_source_region", "hippocampus"),
        target_region=train_config.get("boran_target_region", "entorhinal_cortex"),
    )
    test_loader = data.get("test_loader")
    if test_loader is not None:
        return test_loader
    from torch.utils.data import Subset
    full_ds = data["dataset"]
    ds = Subset(full_ds, test_idx)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
