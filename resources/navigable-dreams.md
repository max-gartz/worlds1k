# Navigable Dreams — Composing Temporal and Spatial Models

The world model predicts *what happens*. A spatial decoder renders *what it looks like*. Composed together: navigable first-person dreams.

## The Two Systems

**Temporal model (this project).** Predicts future world states in latent space, conditioned on current state + actions. Encodes the causal structure of reality: what follows what, how actions change the world, what's likely to happen next. Operates at multiple timescales (frame-level motion through scene-level goals).

**Spatial decoder (future work).** Takes a predicted world state (latent) and renders it from a specific viewpoint. Not a flat frame — a spatially consistent scene you can look around in. Could be a neural radiance field, 3D Gaussian splatting, or a learned point cloud decoder conditioned on the world model's latents.

## How They Compose

```
You: standing in a kitchen, looking at the counter.

World model (temporal):
  z_t     = "kitchen, cup on counter, cabinet closed"
  action  = "reach toward cabinet"
  z_{t+1} = "hand approaching cabinet handle"
  z_{t+2} = "cabinet opening, plates visible inside"

Spatial decoder (per latent):
  z_t + camera_pose → rendered scene you can look around in
  z_{t+1} + camera_pose → updated scene, hand in view
  z_{t+2} + camera_pose → cabinet open, you can look inside
```

At each timestep, the temporal model advances the world state. The spatial decoder renders that state from your current viewpoint. You can move your head, shift your gaze, explore — the spatial decoder handles viewpoint changes within a single world state. When you act (reach, walk, turn), the temporal model advances to the next state.

## Why This Might Work

The vision decoder already proves that level-1 latents contain enough information to reconstruct frames. A 3D-aware decoder trained on the same latents — especially from egocentric/multi-view video where viewpoint variation is in the training data — could reconstruct scenes rather than flat images.

The key is that the world model's latent space doesn't just encode "what the frame looks like" (that would be an image autoencoder). It encodes "what's happening in the world" — object positions, states, dynamics. That's closer to a scene representation than a frame representation, even though it was trained on 2D video.

## What Training Data Enables This

- **Egocentric video** (Ego4D, EPIC-KITCHENS): natural viewpoint variation from head movement. The model learns that turning your head changes what you see but not what exists.
- **Multi-camera datasets** (MultiCamVideo-Dataset): same scene, multiple viewpoints. The latent must encode the scene, not any single view.
- **3D scene datasets** (InternScenes): if rendered from multiple views, provide clean supervision for view-consistent decoding.

## The Phenomenology

This is what dreaming feels like. You're *in* a generated world with spatial extent (look around), temporal evolution (things happen), and causal structure (your actions matter). But nothing is coming from senses — it's all top-down prediction from internal models.

The combination of temporal prediction + spatial rendering is the computational version of that experience.
