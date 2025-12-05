"""
Track B: SpectralShift Fixes
============================

Experiments to improve PSD correction without degrading Stage 1 R².

Available methods:
- PhasePreservingSoftCorrection: Soft beta-power correction preserving phase
- AdaptiveGatedSpectralCorrection: Learned per-sample, per-band gating
- WaveletLocalSpectralCorrection: CWT-based time-local correction
- MultiScaleSpectralLoss: Training-time spectral loss (no block)
- IterativeR2ConstrainedRefinement: Gradient-free iterative correction
"""
