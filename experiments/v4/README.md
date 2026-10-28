# EchoWorld v4 — Action-conditioned Latent World Model

**Status:** Research prototype; genuine PyTorch training, not a reproduction of Google DeepMind's closed-source Genie 4 architecture.

This extension develops v3 from direct pixel-level prediction toward **latent, autoregressive visual dynamics**. Inspired by public Genie/Genie 2 research and Genie 4's published direction of hybrid world models, it trains a small 32×32 RGB encoder, an action-conditioned convolutional recurrent dynamics model, and a decoder.

## What is implemented
- Preserves the **actual v3 EchoWorld simulator** (`src/world.py`, `src/visual.py`), its six actions and state transitions.
- 3-channel image → 12-channel 8×8 latent → convolutional GRU-like dynamics with 6-dimensional action conditioning → predicted latent → decoded RGB image.
- Reconstruction, action-aware change-region loss, and latent regularization.
- Closed-loop 1/10/50-frame prediction; each generated frame becomes the next input.
- Independent episode-seeded test split, frozen-initial-frame baseline, changed-region MSE.
- Visual rollout comparison and a standalone HTML result explorer.

## Preliminary measured results

| Frames ahead | Closed-loop MSE ↓ | Frozen-first-frame MSE ↓ | Changed-region MSE ↓ |
|---|---:|---:|---:|
| 1 | 0.00615 | 0.00312 | 0.04455 |
| 10 | 0.02426 | 0.01466 | 0.02928 |
| 50 | 0.02063 | 0.02285 | 0.03527 |

Setup: 40 train episodes (54 transitions each), 1,500 sampled training pairs, 12 independent test episodes, **5 epochs** on CPU, seed 2026. These are not evidence of convergence. Learned predictions perform **worse than a frozen-frame baseline at 1/10 steps**, while slightly better at 50 steps. At longer horizons, scene saturation and distribution effects complicate simple conclusions. Loss and error are computed over 32×32 simulator-generated images, not natural video.

## Reproduce

```bash
pip install torch numpy pillow
python src/v4.py --episodes 40 --samples 1500 --epochs 5 --seed 2026
python make_dashboard.py
```

Open `index.html`. To improve results, increase episodes and epochs, add balanced high-change sequences and multi-step rollout loss; evaluate over multiple random seeds and with appropriate stronger baselines. See `RESEARCH.md`.

## Project layout

- `src/world.py` — simulator state transitions copied from v3
- `src/visual.py` — synthetic RGB renderer copied from v3
- `src/v4.py` — encoder, latent recurrent dynamics, decoder, dataset, training and rollout evaluation
- `artifacts/v4_latent_world_model.pt` — trained PyTorch weights
- `artifacts/metrics.json` — real measured results
- `artifacts/rollout_comparison.png` — actual multi-step predictions
- `index.html` — standalone visual result dashboard

## Relation to Genie research

**Genie 1 (2024):** spatiotemporal video tokenizer, latent action model, autoregressive dynamics. We implement *known* action-conditioned dynamics and an image autoencoder; **not** its unsupervised latent action discovery or tokenizer architecture.

**Genie 2 (2024):** autoregressive latent diffusion world model. We adopt latent autoregression but **not** diffusion, a large Transformer, classifier-free guidance, or pretrained video backbone.

**Genie 4 (October 2026):** Google describes real-time multimodal simulation, language steering, a Gemini observation/planning loop and an emerging hybrid structured + pixel-level world-model direction. **No public reproduction of the complete Genie 4 training architecture is claimed.** Our work is an independent educational prototype.

References:
- https://arxiv.org/abs/2402.15391
- https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/
- https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/
