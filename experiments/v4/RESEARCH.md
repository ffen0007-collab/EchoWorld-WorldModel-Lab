# v4 Research Note — What It Proves, What It Does Not

## Hypothesis
Will a small recurrent latent world model, rather than a pure RGB next-frame CNN, support persistent action-dependent state changes under long rollouts?

## Method
Train only on simulation frames and **known** action labels. Split by episode seed. Each test rollout starts from a real frame, then uses its own generated images for subsequent predictions. At each horizon compare against the true simulator frame generated under the same action sequence. Use a frozen-initial-frame baseline for context, but note that this weak baseline is particularly competitive when the scene is visually static. The visual encoder is learned end-to-end, not a pretrained VAE. No stochastic generative sampling or audio.

## Observations
The model trains and produces decoded future frames. However, on 12 independent episodes and 5 epochs, short-horizon errors exceed the frozen-frame baseline. One cannot infer reliable world dynamics or controllability from this pilot. Longer horizons can have lower MSE because many object states settle and predicted images regress toward common configurations.

## Key Failure Risks
1. **Exposure bias:** ground-truth input during training, model-generated input during rollout.
2. **Static-pixel dominance:** global MSE can reward unchanged scenery while missing gameplay-critical changes.
3. **Object persistence:** a low MSE cannot guarantee the same tower, bridge or tree survives at stable coordinates.
4. **Action adherence:** scenes may drift or ignore commands; measure counterfactual differences under matched starting frames.
5. **Insufficient independent variation:** synthetic states from one deterministic simulator cannot demonstrate natural-world physics generalization.

## Next scientific experiments
- Matched, multi-seed (≥3) v3 residual vs v4 latent model comparison using the **same data and budget**.
- Scheduled sampling and 4/8-step loss, instead of just one-step loss.
- Balanced transitions that alter `dam`, `grown`, `lit` and weather, plus direct action-conditional accuracy.
- Semantic mask IoU, object identity consistency after leaving/returning, action-conditioned causal success.
- A latent diffusion/flow-based ablation **only** if a simpler recurrent model passes control and persistence tests.
- Optional Gemini planner/verifier as an isolated orchestration layer; must not modify authoritative game state without validation.

## Application to Summer Solstice Train
Re-train on *The Forgotten Hill* state/action schema and its actual renderer. Player actions (repair chime, restore wind, recover audio) should trigger persistent and verifiable scene changes. Emotional narrative stays in a scripted/structured wish engine; the visual model should not author canonical past events. If generation violates state constraints, fall back to deterministic UE5 rendering.

## Responsible Claims
**Not Genie 4**, no access to Genie 4 weights, no exact training-recipe replication, no native speech/audio, no arbitrary natural-language real-time steering, no general 3D world simulation. The code implements selected public ideas and provides falsifiable tests.
