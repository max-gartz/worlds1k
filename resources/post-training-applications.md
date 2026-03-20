# Post-Training Applications

What the model can do once training converges.

---

## 1. Next-State Prediction

Feed a sequence of frames and predict what comes next at every level of the hierarchy. Low levels predict pixel-level motion (edges shifting, textures deforming). Mid levels predict object-level events (a hand reaching for a cup). High levels predict scene-level transitions (the meeting ends, people stand up).

This is the model's native task — inference is just training without the weight update.

## 2. Anomaly Detection

Prediction error is a free anomaly signal, and the hierarchy gives you granularity control. High error at low levels means something visually unexpected (a glitch, a lighting change). High error at mid levels means an unusual object behavior (a car reversing on a highway). High error at top levels means a structurally anomalous event (a person walking into a restricted area during off-hours).

You get a full spectrum of anomaly semantics without training separate detectors.

## 3. Learned Representations

The latent states at each level are compressed, predictive summaries of the visual world. Extract them and use as feature vectors for downstream tasks:

- **Activity recognition** — mid-level states encode action structure naturally.
- **Scene classification** — top-level states capture the slow-moving context (kitchen, intersection, warehouse).
- **Object tracking** — low/mid-level states maintain identity through prediction.
- **Temporal segmentation** — boundary signals (prediction resets) at each level mark transitions at different timescales.

No fine-tuning needed for many tasks — just train a linear probe on top of the frozen representations.

## 4. Learned Action Vocabulary

Each level of the hierarchy develops its own vocabulary of transitions. Low levels learn micro-actions (leftward motion, expansion, rotation). Mid levels learn object actions (pick up, put down, open, close). Top levels learn compound activities (cooking a meal, parking a car).

This vocabulary emerges from prediction, not from labels. Practical uses:

- **Video search** — query by action code rather than keyword.
- **Automatic labeling** — cluster the action codes, assign human-readable names once, then label at scale.
- **Action comparison** — measure similarity between action sequences across different videos.

## 5. Mental Simulation / Planning

Run the predictor forward without new sensory input. Feed a current state, then let each level generate its predicted next state, which feeds back as input to the next step. The model rolls out a plausible future.

This enables counterfactual reasoning: given this scene, what happens if the car accelerates vs. brakes? The model can explore branching futures by conditioning on different action codes at each step.

Key constraint: accuracy degrades over time without error correction from real input. Short-horizon rollouts are reliable; long-horizon ones are speculative. This is a feature, not a bug — it matches how uncertainty actually works.

## 6. Dreaming as Unconstrained Prediction

Cut the sensory input entirely. Initialize from a random or sampled latent state and let the predictor run free.

Top-down flow keeps the trajectory coherent for a while — the high-level state constrains mid-level predictions, which constrain low-level predictions. But without bottom-up error correction, the trajectory drifts. Details mutate. Scenes morph into other scenes. This is structurally identical to what happens in real dreams.

### Why this matters

**Data augmentation.** The model imagines plausible scenarios it has never seen. A warehouse model dreams up novel forklift trajectories, unusual lighting conditions, edge-case near-misses. These synthetic sequences supplement real training data.

**Planning.** Before committing to an action, simulate multiple futures from the current state. Rank outcomes. Pick the best branch. This is model-based reinforcement learning with the world model as the simulator.

**Model consolidation.** Run the dreaming process offline — replay past experiences, let the predictor rehearse and refine its internal model. Consistent sequences get strengthened; inconsistent ones get pruned. This aligns directly with neuroscience theories of sleep function. Hawkins argues that dreaming is the brain rehearsing its predictive models: replaying the day's inputs, reinforcing patterns that held up, and weakening ones that didn't. The model's offline dreaming serves the same purpose — it's unsupervised self-improvement through unconstrained prediction.

## 7. Transfer to Embodied Agents

The trained model already understands visual dynamics — how objects move, how scenes evolve, what actions look like from a third-person or egocentric view. An embodied agent (robot, simulated character) inherits this understanding as a pretrained world model.

What the agent still needs to learn: the mapping from its motor commands to the model's action vocabulary. "When I send torque X to joint Y, the world model sees action code Z." This is a much smaller learning problem than building a world model from scratch.

The transfer path:

1. Pretrain the hierarchical predictor on large-scale video.
2. Deploy on the agent's visual stream — the model predicts what the agent will see next.
3. Train a thin policy layer that maps desired future states to motor commands, using the model's prediction error as the training signal.

The world model does the heavy lifting. The policy just steers.
