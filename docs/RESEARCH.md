# EchoWorld — Research and AI-Native Gameplay Design Case Study

*An independently developed action-conditioned world-model prototype for evaluating persistent and causally meaningful interactions in games.*

**As of 29 October 2026** · [Back to repository overview](../README.md)

## 1. Executive decision

**Product hypothesis:** Open-ended environmental intervention is valuable only when players can anticipate, observe and rely upon lasting consequences. A beautiful but temporally unstable generated scene is insufficient for a puzzle or exploration game that depends on its own history.

**Current evidence:** EchoWorld demonstrates a deterministic game, a learned action-conditioned state model, early visual prediction models and quantitative tests. It does **not yet** demonstrate a full generative AI-native gameplay loop, persistent generated 3D worlds, causal reasoning learned from realistic physics, or a player-study result.

**Development decision:** Adopt a *hybrid* system: use explicit game-state rules as the authority for solvable mechanics; study learned visual/state dynamics where they may provide expressiveness. Introduce open-ended instructions only behind action validation and verification. Evaluate gameplay benefit before scaling model complexity.

## 2. Game experience and player needs

### Setting

On an island whose ecological balance has failed, the player must restore an ancient tower. Weather, moisture, plants, a dam, and a floodgate interact through a simplified cause-and-effect system.

### Jobs to be done and pain points

| Player need | Existing design limitation | Proposed AI-native affordance | Validation question |
|---|---|---|---|
| Experiment, not merely trigger predefined events | A menu of six actions bounds the solution space | Express a goal such as “redirect the current to irrigate this bank” | Do players generate more **valid, distinct** solutions? |
| Trust persistent consequences | A generated object may visually disappear or drift | Maintain stable IDs, state history and consistency checks | Does object/state persistence hold after revisits? |
| Understand consequences | A world may appear to change without explanation | Link observed effects to explicit logged causes | Can players correctly predict subsequent outcomes? |
| Discover without losing a solvable game | Unconstrained generation can violate quest conditions | Validate proposed interventions against mechanical invariants | Are tasks solvable and recoverable after errors? |

These are **design hypotheses**, not measured interview findings.

### Intended core loop

1. **Explore:** identify a blocked route or damaged ecosystem.
2. **Hypothesize:** predict how a change in weather, water or objects could help.
3. **Intervene:** describe or perform an environmental action.
4. **Simulate and verify:** update authoritative world state; render and check its effects.
5. **Reflect and iterate:** use persistent world feedback to refine the plan.

**Example:** “Use fallen wood to make a bridge across the downstream channel.” The system checks whether wood exists, quantity is sufficient, the span can be bridged and the placement is legal. Only then is the bridge registered in world state. A generator can produce varied visual representations without silently changing collision or reachability rules.

### AI-native counterfactual

Following *AI Native Games: A Survey and Roadmap*, ask: **if generative AI were removed or replaced by a trivial fixed script, would the central form of play fundamentally change?**

- **Current v1–v3:** No; these research prototypes evaluate modeling techniques on a fixed set of actions. They are **AI world-model research**, not a fully validated AI-native game.
- **Target game:** Potentially yes, if players' semantically open-ended interventions produce mechanically valid, persistent outcomes that cannot be equivalently reduced to a small action menu.
- **Needed evidence:** a controlled comparison against fixed-action gameplay with matched goals, visuals, and difficulty.

## 3. Frontier research and implementation boundaries

| Research | Publicly supported contribution | What EchoWorld implements | What it does **not** claim |
|---|---|---|---|
| **Genie (2024)** | Learn interactive-environment representations and dynamics from video, including latent-action modeling | Action-conditioned learned transitions | No self-supervised latent-action discovery from internet videos |
| **Genie 2 (2024)** | Action-conditioned generative interactive worlds and image dynamics | Small CNN-based RGB next-frame prediction | No large-scale autoregressive diffusion world model |
| **Genie 4 (27 Oct 2026)** | Real-time text steering, native multimodal simulation, Gemini-guided multi-step observation/planning; hybrid-world-model direction | Product architecture inspired by observe–plan–act–verify and structured/visual separation | No Genie 4 weights, API access, native audio, or official architecture reproduction |
| **AI Native Games survey (2026)** | Generative AI should be constitutive of the core gameplay, not superficial decoration | A counterfactual criterion and player-experience test plan | No completed proof of AI-native player value |

Google DeepMind identifies object drift, spatial and long-term consistency, complex physics and audio precision as ongoing issues for Genie 4. The official page described a limited research preview; model internals were not released for an independent reproduction in this project. See [the official announcement](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/).

## 4. Technical architecture

```text
Player goal / action
       |
       v
Intent parsing / candidate plan        [proposed extension]
       |
       v
Action schema + precondition checks
       |
       v
Authoritative world-state engine       [implemented in simplified form]
       |                         \
       v                          v
Learned state dynamics           Visual dynamics models [implemented]
       |                          |
       +------------+-------------+
                    v
       Observer / constraint verifier   [partly implemented as offline evaluation]
                    |
       accept / retry / fallback        [proposed runtime integration]
```

State dynamics learn `s_(t+1) ≈ f_theta(s_t, a_t)`. Visual dynamics learn `I_(t+1) ≈ g_phi(I_(t-k:t), a_t)`. This factorization is **our experimental design**, not a claimed Genie 4 formula. The authoritative state engine remains the source of gameplay truth, regardless of model prediction.

### Data provenance

The training world is a deterministic, stylized EchoWorld simulator. It produces `(state, action, next_state)` transitions and 32×32 RGB rendered frames. Actions are directly known, so latent-action inference is unnecessary for this controlled prototype. Synthetic labels reflect **simulator rules**, not independent ground truth about real physics.

To reduce leakage, evaluation should split by **whole generated episodes**, not shuffle adjacent frames from a single episode into train and test. v3's reported split uses independent generated episodes, while layouts and simulator logic remain shared.

## 5. Iteration log: v1 → v2 → v3

### v1 — Structured transition prediction

**Question:** Can a small MLP predict world-state changes when given both current state and the player action?

**Implementation:** 10-dimensional state representation; six-way action encoding; action-conditioned MLP; a rule-based simulator for target labels; one-step and critical-path testing.

| Evaluation | Result |
|---|---:|
| Independent test transitions | 4,074 |
| Boolean accuracy | 99.97% |
| Moisture / ecosystem MAE | 1.197 pp |
| Critical-path MAE | 3.57 pp |
| No-change baseline MAE | 5.133 pp |

**Product implication:** average performance alone may be misleading if key puzzle transitions depend on specific threshold crossings.

### v2 — Long-horizon state tests and visual baseline

**Question:** Do one-step gains survive closed-loop feedback and visually meaningful actions?

**State evaluation:** feed each predicted state back as input for subsequent action predictions. Measure endpoints after 10 and 50 actions; separately evaluate boundary combinations.

| Test | Result |
|---|---:|
| 10-step terminal numerical MAE; 100 trajectories | 3.556 pp |
| 50-step terminal numerical MAE; 100 trajectories | 1.13 pp |
| Boundary-set numerical MAE; 2,136 state–action combinations | 21.77 pp |

**Interpretation warning:** 10- and 50-step end metrics are not directly comparable as a monotone stability curve; saturation and trajectory composition confound them. Boundary combinations may include unreachable states and should be distinguished from in-distribution playable episodes.

**Visual baseline:** an action-conditioned CNN predicts the next 32×32 frame. The trained CNN achieved test MSE **0.01585**, versus **0.00341** for simply copying the previous frame. The learned model **underperformed** the cheap persistence baseline.

**Decision:** prioritize residual prediction and dynamic-region evaluation, not model size alone.

### v3 — Residual, change-aware and temporal ablations

**Question:** Can a model improve action-induced changes while retaining stable world details?

**Implemented mechanisms:**

- **Residual prediction:** `next_frame = clamp(previous_frame + 0.5 * tanh(predicted_delta))`.
- **Change-weighted loss:** extra loss weight on pixels with target changes above the training threshold; target change masks are *not available at inference*.
- **Change-rich training sampling:** emphasize visibly changing transitions in train only; leave validation/test unmodified.
- **Four-frame ConvGRU:** encode consecutive observations and predict the next RGB frame conditioned on action.
- **Persistence proxy:** fixed spatial regions and RGB-threshold appearance checks for a dam, illuminated tower, and vegetation, plus region-level error.

**Actual run:** 20 training / 5 validation / 8 test episodes; 586 training, 108 validation, 168 test transitions; 2 epochs. Test images are from the same simplified layout family.

| Model | Total MSE ↓ | Changed-region MSE ↓ |
|---|---:|---:|
| Copy frame | 0.003401 | 0.092203 |
| Standard CNN | 0.019631 | 0.052734 |
| Residual CNN | **0.003361** | 0.089876 |
| Residual + weighted loss | 0.004290 | 0.078301 |
| ConvGRU + weighted loss | 0.006215 | **0.07114** |

![Visual model comparison](../experiments/v3/artifacts/ablation_comparison.png)

**Interpretation:** a tiny absolute overall-MSE advantage for residual prediction is not robust evidence of superiority. The lower changed-region error of ConvGRU+weighting trades off against static-region quality. Changed pixels account for only about 3.69% of test image pixels, making copy-frame an unusually strong global baseline. Two epochs and one run cannot establish convergence or statistical significance.

**Important nuance:** This is not a clean factorial ablation over all independent factors: the ConvGRU row changes architecture, temporal input and weighting jointly. To isolate temporal memory, compare otherwise matched single-frame and multi-frame models at equal parameter or compute budget.

## 6. Proposed model evaluation protocol

| Dimension | Protocol | Metric | Failure / fallback |
|---|---|---|---|
| Action dependence | Compare correct action vs shuffled / ablated action | Per-action delta error, improvement vs no-action baseline | Reject actions whose effect is unreliable |
| Long-horizon consistency | Feed generated frames back over 10/50 frames on held-out action sequences | Horizon-wise frame MSE, LPIPS if available, landmark drift | Periodic re-anchoring from authoritative renderer |
| Object permanence | Track stable IDs across occlusion, camera movement and revisits | ID retention, object-state IoU, false disappearance rate | Restore from registry or deterministic render |
| Causal correctness | Test action sequences and counterfactual interventions | Valid state transitions, outcome constraint violations | Engine-side validation; rollback |
| Gameplay safety | Invalid actions, adversarial prompts, resource depletion | Invalid-action acceptance rate, unsolvable-state incidence | Refuse, clarify, or propose a legal alternative |
| Responsiveness | Measure complete interaction loop, on target hardware | P50/P95 latency, frame-time budget, cost/interaction | Use a small model, asynchronous planning, cached visuals |

**Reporting rules:** use matched unseen episode/layout splits, at least three random seeds where feasible, confidence intervals, and exact compute/latency hardware. Track *both* all-pixel and changed-pixel metrics, plus change mask precision/recall or IoU. Compare against a copied-frame and deterministic-renderer upper-bound reference. Keep **state rollouts** distinct from **video rollouts**.

## 7. Product validation design

### Research questions

- Does generative affordance increase the number of **successful distinct solutions**, rather than merely the number of commands attempted?
- Do players perceive state changes as **causally legible**, persistent, and fair?
- Does model inconsistency reduce trust, agency or puzzle comprehension?

### Controlled comparison (proposal; not yet run)

| Variant | Experience | Purpose |
|---|---|---|
| A — Fixed actions | Six deterministic verbs and authored feedback | Baseline for task completion and understanding |
| B — Open-ended planner + verified state | Players type goals; planner proposes permitted operations | Is semantic openness valuable while maintaining game rules? |
| C — B + learned generative presentation | B's same authoritative state, more expressive model-driven visuals | Is learned presentation worth consistency, latency and compute costs? |

Keep scenario difficulty, hints, objectives, and playtime comparable. Randomize order where feasible and counterbalance learning effects. Start with formative think-aloud sessions; move to a powered study only after verifying instrumentation and a stable prototype.

**Proposed primary outcomes:** unique valid solution paths per task, objective completion, independently scored causal understanding, perceived agency, and frustration/consistency incidents. **Guardrails:** unfinishable quests, broken objects/paths, inappropriate content, P95 latency and monetary cost per session.

### Decision gates

- **Go for B** only if open input produces meaningful additional player strategies without undermining objective completion or causal comprehension.
- **Go for C** only if generative world presentation adds measurable player value beyond B and remains within persistence/latency budgets.
- **No-go / redesign** if the majority of “novel” commands reduce to identical fixed actions, or generated scenes systematically contradict saved state.

No empirical player-study outcome is claimed at the current stage.

## 8. Risk management and scope limits

1. **Hallucinated outcomes:** do not allow generated text/frames to award resources, modify quest flags or create obstacles without accepted state transitions.
2. **Object drift:** persist object IDs, position and material in a registry; monitor landmarks; re-render from the source of truth on mismatch.
3. **Broken causality:** instrument action IDs, preconditions, state diffs and expected effects; reject nonverifiable changes.
4. **Runaway generation cost:** allocate per-interaction latency and token/frame budgets; provide deterministic fallback and graceful loading states.
5. **Unsafe or inappropriate content:** validate player instructions and model outputs; constrain creative operations and provide reporting/muting pathways if user-generated content is introduced.
6. **Overclaiming research:** synthetic simulator accuracy is not evidence of physical generalization; short-term region appearance is not long-term object permanence; current CNN/ConvGRU are not Genie 4.

## 9. Next reproducible experiments

1. **Convergence and seeds:** 15+ epochs, larger training episodes, three seeds; report mean, spread and learning curves. Track total and change-region error jointly.
2. **Factorial ablation:** residual on/off × weighted loss on/off × temporal model on/off, with matched capacity and sampling.
3. **True pixel autoregression:** produce model-generated 10/50-frame rollouts without injecting ground-truth RGB frames after initialization.
4. **Spatial generalization:** modify object placement, map layout, camera and appearance; hold out full configurations.
5. **Semantic object tracking:** benchmark object ID and location continuity from a proper detector/tracker or simulator-provided masks; report proxy and real tracking separately.
6. **Player-facing prototype:** implement bounded free-text interventions, action validation and a causally inspectable replay, then run the controlled A/B/C study.

## 10. References

1. Bruce et al. (2024). [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391).
2. Google DeepMind (2024). [Genie 2: A large-scale foundation world model](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/).
3. Hadsell, Rivas & Qi / Google DeepMind (27 October 2026). [Genie 4: A new generation of world simulation](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/).
4. Xu et al. (2026). [AI Native Games: A Survey and Roadmap](https://arxiv.org/abs/2607.00527).

**Citation policy:** Research ideas are credited to their public sources; implementation choices, small-model experiments, thresholds and the hybrid design are EchoWorld project decisions, not claims about unpublished Genie 4 internals.
