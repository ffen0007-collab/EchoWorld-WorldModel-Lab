# EchoWorld WorldDynamicsModel (v1)

This is a **real trained PyTorch model**, not a mocked model or a Genie 4 implementation. It predicts the next **structured game state** from the current state and one of the six existing EchoWorld actions. The offline HTML dashboard uses exported model weights to run the exact MLP in browser JavaScript.

## Quick start

1. Open `EchoWorld_WorldDynamics_Dashboard.html` in Chrome/Edge/Firefox (works offline). Click actions or the five-step story button.
2. To retrain, install Python 3.10+, `torch` and `numpy`, then run:

```bash
python src/train.py --episodes 1400 --epochs 55 --seed 2026
python src/build_dashboard.py
```

3. Source reference: `EchoWorld_Original_Demo.html`. State transitions in `src/world.py` were ported directly from its `act()` function. Original avatar coordinates, animation frames and UI logs are not machine-learning targets.

## Technical method

- Dataset: independent 5–24-action episodes initialized with the original game state, random actions, and precise rule-engine next states. 80/20 split by episode identifier, not adjacent steps.
- Model: 16 inputs = 10 state features (wet/eco normalized) + six one-hot actions; MLP 16→64→64→10 (ReLU). Two continuous state heads trained with squared error and eight Boolean heads trained via BCEWithLogitsLoss. AdamW, cosine learning rate schedule, seeded run.
- Model predictions in dashboard are one-step and **always from the current real state**; this is not a long-horizon autonomous world rollout. The output rendering is traditional Canvas, not AI-generated video or Genie 4.
- Evaluation: separate held-out episode states, naive no-change baseline and a manually designed five-action critical story path. Story-path repeated samples are not independent.
- The model is a surrogate of existing deterministic rules, not an independently discovered physical world model. Goal: reproducible proof-of-concept to evaluate action-conditioned prediction and failures near gameplay boundaries.

## Limitations / next experiment

- The data are simulated, not real-world or human gameplay data. Repeated randomized paths can make the held-out distribution easier than unseen mechanic combinations.
- The important causal edges (threshold for plant growth, dam+grown unlocking gate) may be underrepresented or poorly approximated. Test them separately and compare against the exact state engine.
- Improve evaluation with boundary-focused examples (wet 44/45, blocked gate), 10/50-step autoregressive rollouts and action distribution shift, reporting task completion separately from per-field accuracy.
- The reference game uses `eco+=60` on growth; because eco begins at 10 and can reach 100, there is no hidden decay mechanic.
