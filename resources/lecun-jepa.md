# LeCun's JEPA: Architecture, Implementations, and Relevance to Hierarchical World Models

Reference document for building a hierarchical predictive world model synthesizing Hawkins + LeCun ideas.

---

## 1. The 2022 Position Paper: "A Path Towards Autonomous Machine Intelligence"

LeCun's June 2022 paper (v0.9.2) proposes a blueprint for autonomous machine intelligence built around a **non-generative, joint-embedding world model**. The core claim: autoregressive token/pixel prediction is insufficient for real intelligence. Instead, machines need world models that predict in abstract representation space.

### The Six-Module Architecture

The paper proposes an agent architecture with six interacting modules:

1. **Configurator** — modulates all other modules based on current task/context (analogous to attention/executive control)
2. **Perception** — estimates world state from sensory input (maps observations → latent state `s`)
3. **World Model** — predicts future world states, possibly conditioned on actions. This is the JEPA core.
4. **Cost Module** — computes energy/cost for a given state (intrinsic + trainable costs)
5. **Short-Term Memory** — stores relevant state history
6. **Actor** — proposes action sequences to minimize predicted cost

The world model is the centerpiece. It takes current state `s[t]` and action `a[t]` and predicts `s[t+1]` — but crucially in **latent space**, not observation space.

### Energy-Based Model (EBM) Formulation

JEPA is formulated as an energy-based model rather than a probabilistic one. For compatible (x, y) pairs, the energy function `F(x, y)` should be low; for incompatible pairs, high. This avoids the normalization problem of probabilistic models and the need to model full distributions over outputs.

The latent variable `z` captures the information about `y` that cannot be predicted from `x` alone — it represents the **residual uncertainty**. During inference, you minimize energy over `z`: `ŷ = argmin_z F(x, y, z)`.

### Why Non-Generative?

LeCun argues generative models (GANs, VAEs, diffusion, autoregressive LLMs) are fundamentally limited because:

- They must model **every detail** of the output (every pixel, every token)
- For high-dimensional continuous spaces (video, robotics), this is intractable
- Most details are irrelevant for planning and decision-making
- Generative models waste capacity on unpredictable noise (leaf textures, water ripples)

JEPA sidesteps this by letting the encoder **discard unpredictable details** and only preserving what's needed for downstream prediction.

## 2. I-JEPA and V-JEPA: What Was Actually Built

### I-JEPA (Image JEPA) — CVPR 2023

The first concrete implementation. Architecture:

- **Context Encoder**: Standard ViT. Processes visible patches of a context block.
- **Target Encoder**: Same ViT architecture, but weights are **EMA (exponential moving average)** of context encoder — not trained via backprop directly. This prevents representation collapse without needing negative samples or contrastive losses.
- **Predictor**: Narrower ViT. Takes context encoder output + positional tokens indicating *where* the target is, outputs predicted target representations.

**Multi-block masking strategy:**
- Sample ~4 target blocks (each 15–20% of image area)
- Sample 1 larger context block (85–100% of image area), removing overlap with targets
- Masking is applied to **encoder output**, not input (unlike MAE which masks input patches)

**Loss:** Average L2 distance between predicted and actual target patch representations.

**Key results:** Learns semantic features without pixel reconstruction. Produces representations that capture object boundaries and semantic segmentation naturally, unlike MAE which learns more textural/low-level features.

**What it is NOT:** I-JEPA is single-scale, single-timestep, image-only. No hierarchy, no temporal prediction, no action conditioning.

### V-JEPA (Video JEPA) — 2024

Extends I-JEPA to spatiotemporal data:

- Masks fixed spatial regions **across multiple frames** (spatiotemporal tubes)
- Predictor fills in missing regions in latent space — learns motion, object interactions, physical dynamics
- Trained entirely self-supervised on unlabeled video
- No pretrained image encoders, no text supervision, no pixel reconstruction
- Demonstrates understanding of object permanence, motion trajectories

### V-JEPA 2 — 2025

Significant scale-up and the first step toward an actual **world model**:

**Two-stage training:**
1. **Actionless pre-training**: >1M hours of video + 1M images. Mask-denoising objective. Learns foundational physics (gravity, permanence, motion) without supervision.
2. **Action-conditioned post-training (V-JEPA 2-AC)**: Post-trains a latent action-conditioned world model using <62 hours of unlabeled robot video (DROID dataset).

**Action-conditioned predictor architecture:**
- 300M-parameter transformer with **block-causal attention**
- 24 layers, 16 heads, 1024 hidden units, GELU activations
- Separate linear input heads for encoded patches, states, and actions
- **Autoregressively predicts next-frame representation** conditioned on action + previous states

This is notable: V-JEPA 2-AC uses **temporal autoregressive prediction** (next-state given action), not masking. The masking approach was for pretraining; the world model itself is autoregressive in latent space.

### Built vs. Theorized — Gap Analysis

| Aspect | 2022 Paper (Theorized) | Actually Built (as of 2025) |
|---|---|---|
| Latent space prediction | ✅ Core of all JEPA variants | ✅ Fully implemented |
| Non-generative training | ✅ Proposed | ✅ I-JEPA, V-JEPA, V-JEPA 2 |
| World model (action-conditioned) | ✅ Proposed | ✅ V-JEPA 2-AC (single scale) |
| Hierarchical JEPA | ✅ Proposed (H-JEPA) | ❌ Not yet built |
| Multi-scale temporal abstraction | ✅ Proposed | ❌ Not yet built |
| Hierarchical planning | ✅ Proposed | ❌ Not yet built |
| Configurator module | ✅ Proposed | ❌ Not yet built |
| Energy-based training at scale | ✅ Proposed | ⚠️ Partially (VICReg-style, not full EBM) |
| Integration with cost/actor | ✅ Proposed | ⚠️ V-JEPA 2-AC does planning, but simple |

**The hierarchical vision is the biggest unfulfilled piece.** Everything built so far is single-scale.

## 3. Latent Space Prediction vs. Pixel Space

This is the foundational insight of JEPA and the most relevant concept for your project.

### The Problem with Pixel Prediction

Consider predicting the next frame of video. A generative model must predict:
- Exact texture of every surface
- Precise leaf positions on a tree swaying in wind
- Water ripple patterns
- Lighting variations

These are **chaotic and unpredictable** at the detail level but **irrelevant** for understanding what's happening. A generative video model wastes massive capacity modeling noise.

### JEPA's Solution

The encoder learns to map observations into a representation space where:
- **Predictable semantic content is preserved** (object positions, velocities, interactions)
- **Unpredictable details are discarded** (textures, noise, exact pixel values)

The predictor then operates entirely in this compressed space. The key equation:

```
s_y = Predictor(Encoder_x(x), z)
```

Where `z` captures residual uncertainty. The target is `Encoder_y(y)`, not `y` itself.

### Why This Matters for World Models

For a hierarchical predictive world model, this means:
- **Lower levels** can have richer representations (more spatial/temporal detail)
- **Higher levels** can have more abstract representations (object identities, relationships, goals)
- Each level only predicts what's predictable **at its level of abstraction**
- No level is forced to reconstruct raw observations

This is fundamentally different from pixel-space world models (e.g., Dreamer, IRIS) which must decode back to observation space, creating a bottleneck.

## 4. Hierarchical JEPA and Comparison to HTM-Style Hierarchies

### LeCun's H-JEPA Proposal

The 2022 paper proposes stacking JEPA modules hierarchically:

- **Level 1 (low-level)**: Operates on short timescales, detailed representations. Predicts milliseconds-to-seconds ahead. Handles fine motor control, immediate sensory prediction.
- **Level 2+**: Operates on longer timescales, increasingly abstract representations. Predicts seconds-to-minutes-to-hours ahead. Handles route planning, goal decomposition.

**Key mechanism:** Higher levels set **subgoals** for lower levels. Planning at the top level produces a coarse action sequence; each coarse action becomes a goal for the level below, which decomposes it into finer actions.

**Information flow:**
- Bottom-up: lower-level representations feed into higher-level encoders
- Top-down: higher-level predictions/goals constrain lower-level predictions

### Comparison: H-JEPA vs. HTM (Hawkins)

| Dimension | H-JEPA (LeCun) | HTM (Hawkins) |
|---|---|---|
| **Core principle** | Predict in learned latent space | Predict in sparse distributed representations (SDRs) |
| **Hierarchy purpose** | Multi-timescale abstraction for planning | Spatial/temporal pattern recognition at multiple scales |
| **Representation** | Dense learned embeddings (ViT features) | Sparse binary vectors (biologically constrained) |
| **Learning** | Offline batch training (SSL) | Online continual learning |
| **Temporal model** | Masking-based pretraining + autoregressive world model | Sequence memory (high-order Markov) |
| **Top-down flow** | Goals/subgoals from planner | Predictions sent down to constrain lower-level inference |
| **Biological fidelity** | Low (engineering-first) | High (cortical column model) |
| **Scale achieved** | V-JEPA 2: SOTA video understanding | HTM: small-scale demos, anomaly detection |
| **Planning** | Explicit (optimize action sequences against world model) | Implicit (prediction = inference, no separate planner) |

### Key Convergences

Both frameworks agree on:
1. **Prediction is central to intelligence** — not reconstruction, not classification
2. **Hierarchy is necessary** — single-scale representations can't handle real-world complexity
3. **Multiple timescales** — low levels = fast/detailed, high levels = slow/abstract
4. **Top-down influence** — higher levels constrain and guide lower levels

### Key Divergences

1. **Representation type**: LeCun uses dense learned embeddings; Hawkins insists on sparse distributed representations. For your project, consider: SDRs have nice properties (robustness, compositionality) but dense embeddings scale better with current hardware.

2. **Online vs. offline learning**: HTM learns continuously from streaming data. JEPA is batch-trained. For a real-time world model, you likely need something closer to HTM's online learning, possibly with JEPA-style objectives.

3. **Separation of prediction and planning**: LeCun separates the world model from the actor/planner. Hawkins treats prediction as an inherent property of every cortical column — no separate planning module. The Hawkins view is more elegant; the LeCun view is more practical for current implementations.

4. **Generative vs. non-generative**: HTM can generate predictions in observation space (it predicts the next SDR input). JEPA explicitly avoids this. For a world model, you may want both: latent-space prediction for planning, with optional decoding for visualization/debugging.

## 5. Masking-Based Prediction vs. Temporal Next-State Prediction

This is a crucial architectural decision for your project.

### Masking-Based (I-JEPA, V-JEPA pretraining)

**How it works:** Take a single observation (image) or short clip (video). Mask out large portions. Predict the masked regions' representations from the visible context.

**Strengths:**
- Learns spatial structure and semantic features very effectively
- Doesn't require temporal sequences — works on static data
- Naturally encourages learning of invariances (the predictor must generalize across mask positions)
- Scale-proven: V-JEPA 2 pretrained on >1M hours of video this way

**Limitations:**
- Doesn't directly model temporal dynamics or causality
- No notion of action or intervention
- Prediction targets are concurrent (same timestep), not future states
- Doesn't learn a forward model of the world

### Temporal Next-State Prediction (V-JEPA 2-AC, classical world models)

**How it works:** Given state `s[t]` and action `a[t]`, predict `s[t+1]`. Autoregressive in time.

**Strengths:**
- Directly models causal dynamics
- Action-conditioned: can simulate "what happens if I do X?"
- Natural basis for planning (rollout trajectories, evaluate costs)
- Aligns with Hawkins' model: the cortex constantly predicts the next input

**Limitations:**
- Requires temporal data (can't learn from static images)
- Error compounds over multi-step rollouts
- Harder to train stably at long horizons

### The V-JEPA 2 Synthesis

V-JEPA 2 uses **both approaches in sequence**:
1. **Pretraining**: Masking-based SSL on massive video data → learns rich spatiotemporal representations
2. **Post-training**: Action-conditioned next-state prediction on robot data → learns causal dynamics

This is likely the right pattern for your project: use masking-based pretraining to learn good representations of your world's structure, then fine-tune with temporal prediction for dynamics and action conditioning.

### Implications for Hierarchical Architecture

At different levels of the hierarchy, you might want different prediction strategies:
- **Lowest level**: Temporal next-state prediction (fine-grained dynamics, fast timescale)
- **Middle levels**: Both masking (spatial structure) and temporal (dynamics at medium timescale)
- **Highest levels**: Longer-horizon temporal prediction (abstract state transitions, goal-level planning)

The masking approach could also be used for **spatial hierarchy** — predicting representations of unseen regions from context at the same timescale, akin to Hawkins' idea that cortical columns predict what adjacent columns are sensing.

---

## Key Takeaways for Implementation

1. **Predict in latent space, not observation space.** This is non-negotiable for scalable world models. Let encoders learn what to discard.

2. **Use EMA target encoders** (from I-JEPA) to prevent representation collapse without contrastive losses. Simpler and more stable than alternatives.

3. **Two-phase training works:** Masking-based pretraining for representations → temporal prediction for dynamics. V-JEPA 2 validates this pipeline.

4. **Hierarchical JEPA is still unbuilt** — this is the frontier. LeCun proposed it but Meta hasn't shipped it. If you build multi-scale temporal abstraction with JEPA-style latent prediction, you're in novel territory.

5. **Bridge Hawkins and LeCun** by:
   - Using JEPA's latent-space prediction framework (proven, scalable)
   - Incorporating HTM's emphasis on online/continual learning
   - Adopting HTM's bidirectional information flow (predictions down, errors up)
   - Considering sparse representations at higher levels for compositionality

6. **The action-conditioned predictor design from V-JEPA 2-AC** (300M transformer, block-causal attention, autoregressive next-frame prediction) is a concrete starting point for the dynamics model at each level.

---

## Sources

- [LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [Meta AI: I-JEPA Blog Post](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
- [Meta AI: V-JEPA Blog Post](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
- [Meta AI: V-JEPA 2 Blog Post](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- [V-JEPA 2 Paper (arXiv)](https://arxiv.org/abs/2506.09985)
- [I-JEPA Paper (CVPR 2023)](https://arxiv.org/abs/2301.08243)
- [I-JEPA GitHub](https://github.com/facebookresearch/ijepa)
- [V-JEPA GitHub](https://github.com/facebookresearch/jepa)
- [V-JEPA 2 GitHub](https://github.com/facebookresearch/vjepa2)
- [Critical Review of LeCun's JEPA Paper (Malcolm Lett)](https://malcolmlett.medium.com/critical-review-of-lecuns-introductory-jepa-paper-fabe5783134e)
