# EchoWorld v2 — Autoregressive & Visual World Dynamics

## Overview

EchoWorld v2 extends the structured world dynamics model introduced in v1 by investigating long-horizon prediction, boundary-state robustness, and action-conditioned visual generation.

The project explores whether learned world models can reliably simulate persistent environmental changes in AI-native games.

## Research Objectives

1. Evaluate model stability over 10-step and 50-step autoregressive rollouts.
2. Identify failure cases near critical gameplay state boundaries.
3. Train an action-conditioned visual dynamics model.
4. Compare learned visual prediction against a frame-copy baseline.

## Technical Implementation

### Autoregressive World Dynamics

The model repeatedly uses its own predicted state as input for subsequent predictions, allowing evaluation of accumulated prediction errors.

### Boundary-State Evaluation

The evaluation includes extreme or uncommon world states to investigate generalization beyond ordinary gameplay trajectories.

### Action-conditioned Visual Dynamics

A PyTorch CNN predicts the next RGB frame using the previous observation and player action.

The visual model operates on 32 × 32 RGB frames.

## Experimental Results

| Metric | Result |
|---|---:|
| 10-step endpoint numerical MAE | 3.56 percentage points |
| 50-step endpoint numerical MAE | 1.13 percentage points |
| Boundary-state numerical MAE | 21.77 percentage points |
| Visual CNN test MSE | 0.01585 |
| Frame-copy baseline MSE | 0.00341 |

These results come from preliminary experiments and should not be interpreted as evidence of model convergence.

The lower 50-step endpoint error does not necessarily imply better long-term predictive accuracy, because state saturation and trajectory composition can affect endpoint metrics.

## Key Findings

- Strong one-step performance does not guarantee boundary-state robustness.
- Long-horizon evaluation is essential for understanding world-model reliability.
- The initial visual CNN underperformed a simple frame-copy baseline.
- Further research should focus on residual prediction, change-aware loss functions, and temporal modeling.

## Transition to v3

The findings from v2 motivated the following improvements:

- Residual / delta prediction
- Change-region weighted loss
- Change-enriched training samples
- Object-persistence evaluation
- Multi-frame ConvGRU
- Quantitative ablation studies

## Limitations

EchoWorld v2 uses a simplified synthetic environment. It does not implement Genie 4 or reproduce its unpublished architecture.

The experiments investigate selected world-model concepts inspired by publicly available research.