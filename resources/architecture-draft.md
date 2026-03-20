# Hierarchical Predictive World Model — Architecture Draft

## Core Idea

A multi-level latent-space model that jointly learns sensory encodings and action representations from video, driven entirely by prediction error. Each level operates at a progressively coarser timescale: level 1 predicts frame-to-frame, level 2 predicts over short sequences, level 3 over episodes. Higher levels send top-down priors that constrain lower-level predictions.

---

## 1. Notation & Components

| Symbol | Meaning |
|--------|---------|
| $x_t$ | Raw sensory frame at time $t$ |
| $z_t^{(\ell)}$ | Latent state at level $\ell$, time $t$ |
| $a_t^{(\ell)}$ | Inferred action/transition code at level $\ell$ |
| $E^{(\ell)}$ | Encoder from level $\ell{-}1$ representations to level $\ell$ |
| $P^{(\ell)}$ | Predictor (transition model) at level $\ell$ |
| $D^{(\ell)}$ | Top-down decoder from level $\ell$ to level $\ell{-}1$ |
| $T_\ell$ | Temporal stride of level $\ell$ (how many level-$(\ell{-}1)$ steps per one level-$\ell$ step) |

Level 0 is the pixel level: $z_t^{(0)} = x_t$.

## 2. The Core Loop at a Single Level

At every level $\ell$, the same logic repeats:

**Encode.** Compress the sequence of lower-level states into a latent:
$$z_t^{(\ell)} = E^{(\ell)}\!\left(z_{t-T_\ell+1}^{(\ell-1)},\; \ldots,\; z_t^{(\ell-1)}\right)$$

**Infer action.** The transition code $a_t^{(\ell)}$ captures *what changed* between consecutive latent states at this level:
$$a_t^{(\ell)} = f_a^{(\ell)}\!\left(z_t^{(\ell)},\; z_{t+T_\ell}^{(\ell)}\right)$$

This is learned jointly — $a_t^{(\ell)}$ is a bottleneck that must be informative enough to support prediction but compact enough to generalize. It is *not* a motor command; it's whatever abstract transition descriptor the level needs.

**Predict.** Given current state and action, predict next state, biased by top-down context from the level above (see §5):
$$\hat{z}_{t+T_\ell}^{(\ell)} = P^{(\ell)}\!\left(z_t^{(\ell)},\; a_t^{(\ell)}\right) + \alpha \cdot c_t^{(\ell+1)}$$

where $c_t^{(\ell+1)} = D^{(\ell+1)}(z_k^{(\ell+1)})$ is the top-down context and $\alpha$ is a learned scalar.

**Learn from error:**
$$\mathcal{L}^{(\ell)} = \left\| z_{t+T_\ell}^{(\ell)} - \hat{z}_{t+T_\ell}^{(\ell)} \right\|^2 + \lambda_s \, \Omega\!\left(z_t^{(\ell)}\right) + \lambda_a \, \Omega\!\left(a_t^{(\ell)}\right)$$

The $\Omega$ terms are sparsity penalties (see §7). The prediction error drives all learning — there is no reconstruction loss back to pixels except implicitly through the hierarchy.

## 3. Hierarchical Timescales — The Key Mechanism

This is the crux. How does level 2 "operate slower" than level 1?

### The mechanism: temporal striding with learned compression

Each level $\ell$ consumes $T_\ell$ steps of the level below and produces **one** latent state. Concretely:

**Level 1** ($T_1 = 1$): Encodes each frame independently.
- Input: single frame $x_t$
- Output: $z_t^{(1)}$ — a spatial feature map (think: "what objects are where right now")
- Predicts: $z_{t+1}^{(1)}$ — next frame's features
- Timescale: ~33ms (one frame at 30fps)

**Level 2** ($T_2 = 8$): Encodes 8-frame chunks.
- Input: $z_t^{(1)}, z_{t+1}^{(1)}, \ldots, z_{t+7}^{(1)}$ — a short temporal window
- Encoder: a small temporal transformer or 1D conv over the sequence of level-1 latents, producing one vector $z_k^{(2)}$ where $k = \lfloor t/8 \rfloor$
- Output: $z_k^{(2)}$ — captures the *gist* of what happened in that 8-frame chunk ("ball moved left," "hand reached for cup")
- Predicts: $z_{k+1}^{(2)}$ — what the *next* 8-frame chunk will look like in aggregate
- Timescale: ~250ms

**Level 3** ($T_3 = 8$ relative to level 2, so $8 \times 8 = 64$ frames): Encodes 64-frame episodes.
- Input: 8 consecutive level-2 latents
- Output: $z_m^{(3)}$ — the "scene summary" ("person is making coffee")
- Predicts: $z_{m+1}^{(3)}$ — what the next ~2 seconds will be about
- Timescale: ~2s

### What the encoder actually does

The encoder $E^{(\ell)}$ is a **learned temporal pooling** operation. It is *not* just mean-pooling — it's a small network (e.g., causal transformer with $T_\ell$ positions) that can attend to the most informative moments in the window. The key constraint is that it must produce a **fixed-size** output regardless of input length, which forces compression.

The stride $T_\ell$ can be fixed (simplest), or learned via a gating mechanism that decides when a level-$\ell$ state is "complete" and should be emitted. Fixed strides are easier to implement and parallelize; learned segmentation is more powerful but harder to train stably. **Start with fixed strides.**

### Why this works

Level 2 doesn't literally "run slower" — it runs on every frame, but its *state only updates* every $T_2$ frames. Between updates, level 2's current state serves as a stable context for level 1. This creates a natural separation: level 1 handles fast dynamics (motion, flicker), level 2 handles medium dynamics (actions, short interactions), level 3 handles slow dynamics (goals, scene structure).

## 4. Joint Action-Sensory Learning

Actions are not provided as external labels. Instead, action codes $a_t^{(\ell)}$ are **inferred** as the minimal information needed to predict the next state given the current state:

$$a_t^{(\ell)} = f_a^{(\ell)}\!\left(z_t^{(\ell)},\; z_{t+T_\ell}^{(\ell)}\right) \quad \text{(encoder trained end-to-end)}$$

At level 1, $a_t^{(1)}$ might learn to represent optical flow or local motion vectors. At level 2, $a_k^{(2)}$ might represent "reach," "push," or "turn." At level 3, $a_m^{(3)}$ might represent "go to kitchen" or "start new task."

The action space is not predefined — it emerges from the prediction bottleneck. If an action code carries more information than needed, the sparsity penalty compresses it. If it carries too little, prediction error increases and the model expands its use of the code.

When real motor commands are available (e.g., from an embodied agent), they can be fed as an additional input to $P^{(\ell)}$ at the lowest level, and the inferred $a_t^{(1)}$ learns to align with them.

## 5. Top-Down Prediction Flow

Higher levels constrain lower levels via the context signal $c_t^{(\ell+1)}$:

$$c_t^{(\ell+1)} = D^{(\ell+1)}\!\left(z_k^{(\ell+1)}\right)$$

where $k$ is the current level-$(\ell{+}1)$ time index. This is a learned projection (linear or one-layer MLP) from the higher-level state into the same dimensionality as the level-$\ell$ latent.

The predictor at level $\ell$ receives this as a bias or gating signal:

$$\hat{z}_{t+T_\ell}^{(\ell)} = P^{(\ell)}\!\left(z_t^{(\ell)},\; a_t^{(\ell)}\right) + \alpha \cdot c_t^{(\ell+1)}$$

The additive form is simplest. Multiplicative gating ($\odot$ instead of $+$) gives the higher level more control but is harder to train. Start additive.

**Effect:** When level 3 has confidently predicted "person continues making coffee," this biases level 2's predictions toward coffee-making sub-actions, which biases level 1 toward the corresponding motion patterns. Surprise at any level propagates up as prediction error.

## 6. Full Training Objective

$$\mathcal{L} = \sum_{\ell=1}^{L} \gamma_\ell \, \mathbb{E}_t\!\left[\left\| z_{t+T_\ell}^{(\ell)} - \hat{z}_{t+T_\ell}^{(\ell)} \right\|^2\right] + \lambda_s \sum_{\ell=1}^{L} \mathbb{E}_t\!\left[\Omega\!\left(z_t^{(\ell)}\right)\right] + \lambda_a \sum_{\ell=1}^{L} \mathbb{E}_t\!\left[\Omega\!\left(a_t^{(\ell)}\right)\right]$$

The weights $\gamma_\ell$ balance levels. In practice, higher levels have fewer time steps, so per-step weighting with $\gamma_\ell = 1$ often suffices. All encoders, predictors, decoders, and action inference networks are trained jointly end-to-end via this single objective.

## 7. Sparsity / SDR Constraints

Without explicit constraints, the model can collapse: all latents converge to the same point, prediction error goes to zero trivially, and nothing is learned.

The sparsity penalty $\Omega$ enforces **sparse distributed representations**:

$$\Omega(z) = \left| \frac{\|z\|_1}{d} - \rho \right|^2$$

where $d$ is the latent dimension and $\rho$ is a target activation fraction (e.g., $\rho = 0.05$ means ~5% of units active). This encourages:

- **Sparsity**: most units are near zero, preventing the "everything is average" collapse
- **Distribution**: the *which* 5% varies across inputs, giving exponential representational capacity ($\binom{d}{k}$ possible patterns for $k$ active units out of $d$)
- **Disentanglement**: sparse codes tend to separate independent factors because overlapping codes are penalized

An alternative (or complement) is to use a **top-k activation function** after the encoder: keep the $k$ largest activations, zero the rest. This is non-differentiable but works with straight-through estimators and is simpler to tune than the soft penalty.

---

## Summary of Data Flow

```
Frame x_t
  │
  ▼
E¹: encode single frame → z_t^(1)        [every frame, ~33ms]
  │
  ├─→ P¹: predict z_{t+1}^(1) from z_t^(1), a_t^(1), c^(2)
  │         loss: ‖z_{t+1}^(1) - ẑ_{t+1}^(1)‖²
  │
  ▼ (every 8 frames)
E²: encode 8 z^(1)'s → z_k^(2)           [every 8 frames, ~250ms]
  │
  ├─→ P²: predict z_{k+1}^(2) from z_k^(2), a_k^(2), c^(3)
  │         loss: ‖z_{k+1}^(2) - ẑ_{k+1}^(2)‖²
  │
  ├─→ D²: project z_k^(2) → c^(2) for level 1  [top-down]
  │
  ▼ (every 64 frames)
E³: encode 8 z^(2)'s → z_m^(3)            [every 64 frames, ~2s]
  │
  ├─→ P³: predict z_{m+1}^(3)
  ├─→ D³: project z_m^(3) → c^(3) for level 2  [top-down]
```

All arrows carry gradients. The whole thing trains end-to-end from video with one loss: predict the next latent state at every level.
