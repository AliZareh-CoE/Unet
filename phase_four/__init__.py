"""Phase 4: Biological Validation of Neural Translation.

Trains CondUNet on each dataset with a 70/30 train/test split, generates
synthetic (predicted) target signals on the held-out test set, and runs
three families of biological validation:

1. Spectral fidelity  – PSD match, per-band R², CFC (PAC) preservation
2. CCA / DCCA         – canonical-correlation baselines vs UNet
3. Single-trial decoding – classifier trained on real targets, tested on
                           predicted targets (and vice-versa)

Generated signals are persisted to ``/data/synth/<dataset>/`` so downstream
analyses can be run independently of the training step.
"""
