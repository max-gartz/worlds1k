# Decoder Training — Phase 2: Latent → Pixel

## Why a Separate Phase

The core model (Phase 1) learns purely in latent space — every loss is $\|z - \hat{z}\|^2$, never $\|x - \hat{x}\|^2$. This is deliberate: adding a pixel reconstruction objective during Phase 1 would pressure the encoder $E^{(1)}$ to retain pixel-level detail (textures, exact colors) at the expense of learning abstract, predictively useful features. Keeping the phases separate means the latent space is shaped entirely by prediction, not by reconstruction fidelity.

Phase 2 trains a standalone decoder $D_\text{pixel}$ that maps latent states back to frames. The core model's weights are frozen — the decoder learns to read the representations, not reshape them.

---

## Architecture

The decoder maps level-1 latent states to pixel frames:

$$\hat{x}_t = D_\text{pixel}\!\left(z_t^{(1)}\right)$$

Three options in ascending quality:

| Approach | Description | Tradeoff |
|----------|-------------|----------|
| **Transposed CNN** | Stack of ConvTranspose2d layers upsampling $z_t^{(1)}$ to image resolution | Fast, simple; tends toward blurry outputs |
| **Lightweight diffusion** | Small denoising network conditioned on $z_t^{(1)}$; 4–8 diffusion steps at inference | Sharper; moderate compute; handles multimodal uncertainty well |
| **Pretrained VAE decoder** | Take a frozen image decoder (e.g., Stable Diffusion's VAE decoder) and fine-tune a small adapter that projects $z^{(1)}$ into the VAE's expected latent format | Best visual quality; leverages billions of images of pretraining; heavier |

**Start with the transposed CNN** for fast iteration. Move to diffusion or VAE-adapter once the core model's representations are stable.

---

## Training Objective

Given a dataset of $(x_t, z_t^{(1)})$ pairs (frames and their frozen encoder outputs):

$$\mathcal{L}_\text{decoder} = \underbrace{\left\| x_t - \hat{x}_t \right\|^2}_{\text{MSE}} + \lambda_p \underbrace{\sum_l \left\| \phi_l(x_t) - \phi_l(\hat{x}_t) \right\|^2}_{\text{perceptual (VGG features)}} + \lambda_\text{adv} \underbrace{\mathcal{L}_\text{GAN}(x_t, \hat{x}_t)}_{\text{adversarial}}$$

where $\phi_l$ are intermediate VGG-19 features at layer $l$.

**Recommended recipe:** perceptual + MSE first (stable, sharp enough for debugging). Add adversarial term later if visual quality matters for downstream use.

---

## Frozen Encoder — Critical Constraint

During Phase 2, **all** Phase 1 components are frozen:

- Encoders $E^{(1)}, E^{(2)}, E^{(3)}$ — frozen
- Predictors $P^{(\ell)}$ — frozen
- Top-down decoders $D^{(\ell)}$ — frozen
- Action inference $f_a^{(\ell)}$ — frozen

Only $D_\text{pixel}$ (and the optional discriminator) receive gradients. This is non-negotiable: allowing gradients to flow back through $E^{(1)}$ would distort the latent space to make reconstruction easier, undermining the prediction-only objective from Phase 1.

---

## Applications Unlocked

With a trained $D_\text{pixel}$, the system gains visual output capabilities:

**Predicted-frame rendering.** Given $z_t^{(1)}$, predict $\hat{z}_{t+1}^{(1)}$ via $P^{(1)}$, then decode: $\hat{x}_{t+1} = D_\text{pixel}(\hat{z}_{t+1}^{(1)})$. Directly visualizes what the model expects to see next.

**Dream mode.** Run $P^{(1)}$ autoregressively for $N$ steps with no sensory input, decoding each step to a frame. Produces a "dreamed" video sequence — reveals the model's internal world dynamics.

**Navigable world exploration.** Feed a sequence of action codes $a_0, a_1, \ldots$ into the predictor, decode each resulting latent. Allows interactive exploration of the model's learned environment.

**Debugging & interpretability.** Decode arbitrary latent states (including top-down priors $c^{(2)}, c^{(3)}$ projected into level-1 space) to visualize what the model represents at each level.

---

## Multi-Resolution Decoding

Optionally, train separate decoders from higher levels:

$$\hat{x}_t^{(\ell)} = D_\text{pixel}^{(\ell)}\!\left(z_k^{(\ell)}\right)$$

| Level | Expected output | Use case |
|-------|----------------|----------|
| $z^{(1)}$ | Sharp frame (full detail) | Primary decoder |
| $z^{(2)}$ | Blurry scene layout (objects, rough arrangement) | Visualize what the 250ms-level "sees" |
| $z^{(3)}$ | Abstract scene type (color blobs, spatial structure) | Visualize episode-level understanding |

Higher-level decoders will necessarily produce lower-fidelity images — $z^{(3)}$ discards fine detail by design. That's the point: seeing what each level retains reveals the information hierarchy.
