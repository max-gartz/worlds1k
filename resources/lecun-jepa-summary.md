# LeCun JEPA — Concise Summary for Hierarchical World Model Design

Quick reference distilled from LeCun's 2022 position paper and Meta's I-JEPA / V-JEPA implementations. Focuses on what matters for building a hierarchical predictive world model.

---

## 1. The Core Idea: Predict in Latent Space, Not Pixel Space

The central thesis of "A Path Towards Autonomous Machine Intelligence" (LeCun, June 2022) is that generative models — including autoregressive LLMs and diffusion models — are fundamentally limited because they must predict every detail of their output. For high-dimensional continuous data like video, most of those details (leaf textures, water ripples, lighting noise) are chaotic and irrelevant to understanding.

JEPA (Joint Embedding Predictive Architecture) solves this by having two encoders map inputs x and y into a shared representation space, then training a predictor to map the representation of x to the representation of y. The encoders learn to discard unpredictable details and preserve only what's semantically useful. The predictor never touches raw observations — it operates entirely in latent space.

This is the single most important design principle: **the world model predicts abstract states, not observations.**

## 2. The Six-Module Agent Architecture

LeCun proposes six interacting modules:

- **Perception**: maps sensory input to latent state s
- **World Model** (JEPA core): given s[t] and action a[t], predicts s[t+1] in latent space
- **Cost Module**: scores states (intrinsic drives + learned objectives)
- **Actor**: proposes action sequences that minimize predicted cost over the world model
- **Short-Term Memory**: retains recent state history
- **Configurator**: modulates all other modules based on current task/context (executive control)

The world model is formulated as an energy-based model (EBM): compatible (x, y) pairs have low energy, incompatible pairs have high energy. A latent variable z captures residual uncertainty — the information about y not predictable from x alone.

## 3. Hierarchical JEPA (H-JEPA)

The most architecturally relevant proposal, and the part Meta has **not yet built** as of early 2026.

H-JEPA stacks JEPA modules at multiple levels. Each level operates at a different timescale and abstraction level:

- **Level 1** (low): short timescales, spatially detailed representations. Predicts milliseconds-to-seconds ahead. Handles fine motor control and immediate sensory prediction.
- **Level 2+** (higher): longer timescales, increasingly abstract representations. Predicts seconds-to-minutes-to-hours. Handles goal decomposition and planning.

**Key mechanism — hierarchical planning via subgoals:** Higher levels produce coarse action plans. Each coarse action becomes a goal for the level below, which decomposes it into finer sub-actions, recursively down to motor commands.

**Information flow is bidirectional:**
- Bottom-up: lower-level representations feed into higher-level encoders
- Top-down: higher-level predictions/goals constrain lower-level predictions

This mirrors Hawkins' cortical hierarchy (predictions flow down, prediction errors flow up) but uses dense learned embeddings rather than sparse distributed representations.

## 4. I-JEPA (Image JEPA) — CVPR 2023

The first concrete JEPA implementation. Architecture:

- **Context encoder** (ViT): processes visible image patches
- **Target encoder** (same ViT, EMA-updated weights): encodes target patches. EMA update prevents representation collapse without needing contrastive losses or negative samples.
- **Predictor** (narrow ViT): takes context encoder output + positional tokens for target location, predicts target representations

Training uses multi-block masking: ~4 target blocks (15-20% of image each), 1 context block (85-100%). Loss is L2 between predicted and actual target representations.

**Key result:** Learns semantic features (object boundaries, segmentation) without pixel reconstruction. Produces higher-level features than MAE, which tends toward textural/low-level features.

**Limitation for world models:** Single-scale, single-timestep, image-only. No hierarchy, no temporal prediction, no action conditioning.

## 5. V-JEPA (Video JEPA) — 2024

Extends I-JEPA to spatiotemporal data:

- Masks spatiotemporal tubes (fixed spatial regions across multiple frames)
- Predictor fills missing regions in latent space — learns motion, object interactions, physical dynamics
- Entirely self-supervised on unlabeled video, no pixel reconstruction
- Learns implicit physics: object permanence, motion trajectories

## 6. V-JEPA 2 — 2025

The first step toward an actual world model. Two-phase training:

1. **Actionless pretraining**: >1M hours of video + images. Mask-denoising objective learns foundational physics without supervision.
2. **Action-conditioned post-training (V-JEPA 2-AC)**: Post-trains a latent action-conditioned world model using <62 hours of unlabeled robot video (DROID dataset).

The action-conditioned predictor is a 300M-parameter transformer with block-causal attention, autoregressively predicting next-frame representations given action + previous states. Notably, this phase is **temporally autoregressive** (next-state given action), not masking-based.

V-JEPA 2-AC achieves zero-shot robot control in new environments — the first demonstration of a JEPA-based world model enabling planning and action.

**Still single-scale.** The hierarchical multi-timescale vision from the 2022 paper remains unimplemented.

## 7. What's Built vs. What's Theorized

| Aspect | Status |
|---|---|
| Latent-space prediction | Built (all JEPA variants) |
| Non-generative training | Built |
| Action-conditioned world model | Built (V-JEPA 2-AC, single scale) |
| Hierarchical JEPA (H-JEPA) | **Not built** |
| Multi-scale temporal abstraction | **Not built** |
| Hierarchical planning via subgoals | **Not built** |
| Configurator module | **Not built** |

## 8. Key Design Principles for a Hierarchical Predictive World Model

From the JEPA line of work:

1. **Predict in latent space.** Let encoders learn what to discard. No level should reconstruct raw observations.
2. **Use EMA target encoders** to prevent collapse. Simpler and more stable than contrastive methods.
3. **Two-phase training works:** masking-based pretraining for representations, then temporal next-state prediction for dynamics. V-JEPA 2 validates this pattern.
4. **H-JEPA is open territory.** Building multi-scale temporal abstraction with JEPA-style latent prediction is the unfulfilled piece of LeCun's vision.
5. **Masking for spatial structure, autoregression for temporal dynamics.** Different levels of the hierarchy may benefit from different prediction strategies — masking at lower levels for rich spatial features, autoregressive prediction at higher levels for causal dynamics and planning.

---

## Sources

- [LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [I-JEPA Paper (CVPR 2023)](https://arxiv.org/abs/2301.08243)
- [Meta AI: I-JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
- [Meta AI: V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
- [V-JEPA 2 Paper](https://arxiv.org/abs/2506.09985)
- [Meta AI: V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- [GitHub: facebookresearch/jepa](https://github.com/facebookresearch/jepa)
