# EchoWorld v3: action-conditioned visual world dynamics

This project extends EchoWorld v2. **All predictions are real trained PyTorch models** and the data source is a stylized 32×32, deterministic game-state renderer. It is **not a Genie 4 implementation**.

## Reproduce

```bash
pip install torch numpy pillow
python src/upgrade.py --epochs 2 --train-episodes 20 --val-episodes 5 --test-episodes 8
python make_dashboard.py
```

Increase to `--epochs 15 --train-episodes 200 --val-episodes 30 --test-episodes 40` for a more meaningful (CPU-intensive) study. Do not compare measurements generated under different configurations directly.

## Implemented

- **Residual delta prediction:** `frame_next = clamp(frame_prev + 0.5*tanh(CNN(frame_prev,action)))`.
- **Action-change-weighted pixel loss:** `weight=1+7*I(max_RGB_change > 0.025)` using the true training target only; never use this mask as an inference input.
- **Change-rich sampling:** duplicate transition samples with visible changes in the training split, while leaving validation/test unmodified. True observed changed fraction is logged.
- **Four-frame ConvGRU:** encode 4 frames with 3×3 CNN, spatial GRU state, concatenate action planes, residual RGB output.
- **Object appearance / persistence proxy:** fixed semantic image regions and RGB threshold-based appearance detection for dam, illuminated tower, grown plants, plus per-region RGB MSE. This is not general object detection and does not prove long-horizon object permanence.
- **Quantitative ablation:** copy, original CNN, residual, residual+weighting, ConvGRU+weighting. Same test episodes.

## Important interpretation

Small sample run: 20 training episodes and 2 epochs. A lower changed-region MSE at the expense of unchanged-region MSE is **not a global victory**. Main limitations: limited actions and variation, deterministic renderer, identical layout across all episodes, no multi-step autoregressive video rollout, no repeated seeds/confidence intervals. These are future studies, not delivered claims.

## Files
- `src/world.py`: original game world transitions
- `src/visual.py`: deterministic renderer and original CNN
- `src/upgrade.py`: new dataset, architectures, training, metrics, figures
- `artifacts/*pt`: trained weights, load only from trusted sources
- `artifacts/ablation_results.json`: per-variant metrics and history
- `EchoWorld_v3_Visual_Lab.html`: self-contained interactive comparison viewer (exported predictions)
