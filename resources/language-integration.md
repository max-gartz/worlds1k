# Language Integration — Early Thinking

How language fits into the hierarchical predictive world model. Not an implementation plan — just the conceptual direction.

## Language Is Not a Modality

There is no language receptor. Language is a *code* that arrives through actual sensory modalities — speech through audition, text through vision. The brain doesn't have a language input channel; it has ears and eyes, and language is a pattern that those channels carry.

This means the principled approach is **not** "add a text encoder as a third stream." A frozen CLIP encoder that ingests raw text bypasses the sensory hierarchy entirely — it's a shortcut that injects pre-extracted semantics rather than learning to extract them from sensory input.

## What the Hierarchy Should Do Instead

The model already has a visual stream (DINOv2) and an audio stream (Whisper). If the training data includes narrated video:

- The **visual stream** sees a person opening a cabinet.
- The **audio stream** hears "I'm opening the cabinet."

Both describe the same underlying world state. The hierarchy's job — especially at levels 2-3 where timescales match sentence length — is to discover that these co-occur and compress them into the same latent region. Language-level concepts should **emerge** from the sensory prediction objective, not be injected from outside.

This is the Hawkins view: language areas in the cortex are not walled off from sensory processing. They sit at the top of the same hierarchy, operating on the same predictive principles, learning compressed representations of the same world.

## The Whisper Path

Whisper is interesting here because it was trained on speech. Its encoder doesn't just hear "audio" — it extracts representations that are already structured around linguistic content (phonemes, words, prosody). When we use Whisper as the audio backbone:

1. Level 1 audio features capture acoustic events (sounds, speech onset).
2. Level 2 audio features (~250ms) should capture word-level content.
3. Level 3 audio features (~2s) should capture sentence/utterance-level meaning.

Meanwhile the visual hierarchy at those same levels captures frame dynamics, actions, and scene structure. The prediction objective forces these to become mutually predictive: level 3 can't predict the next visual state well without knowing what was just said, and vice versa.

**If this works, language understanding emerges from audiovisual prediction. No text encoder needed.**

## Where Text Encoders Might Still Help

The pure sensory path is principled but data-hungry. Two pragmatic shortcuts:

**Contrastive bootstrapping.** Use a text encoder (CLIP/SigLIP) to provide alignment targets during training — a contrastive loss that pulls level-2/3 latents toward the text embedding of the narrated action. This doesn't add language as a modality; it provides a training signal that accelerates convergence toward representations that happen to be language-aligned. The text encoder can be removed after training.

**Evaluation.** Even if language understanding emerges from sensory prediction, we need a way to probe what the model learned. Text embeddings provide a measurement tool: how close is the model's level-3 latent for a "making coffee" scene to the CLIP embedding of "making coffee"? This is evaluation, not architecture.

## What Data Enables This

Narrated video is the key. Datasets where someone describes what they're doing while doing it:

- **EPIC-KITCHENS** — ~40K narrated action segments ("open door", "take plate") with audio + video
- **Ego4D** — egocentric video with narrations and audio
- **FineVideo** — YouTube videos with speech-to-text transcriptions

The narrations don't need to be input features. They just need to be *present in the audio stream*. If the video has sound and the sound includes speech, the hierarchy has everything it needs.

## Text as a Control Interface

There is a principled engineering path that doesn't conflate language with perception: use a text encoder as a **control interface** into the latent space.

The idea: once the hierarchy has learned rich latent representations from audiovisual prediction, build a learned projection from text embeddings into level-2/3 latent space. This projection maps "pick up the red cup" into the region of latent space that the model associates with that action — allowing text to steer the world model's predictions.

This is not adding language as a modality. It's building a remote control. The model was trained on sensory prediction; the text projection is a way to inject semantic intent without going through the sensory pipeline. It's the same pattern as how a robot policy maps motor commands into the action code space — an interface for control, not a perceptual stream.

The important property: **the text projection is trained after or alongside the world model, but it doesn't participate in the prediction objective.** The latent space is shaped entirely by sensory prediction. The text encoder just learns to point into it.

This means:
- The same latent space will eventually be reachable by audio (speech through Whisper) once the audio pathway is strong enough.
- Text input is a faster/cheaper engineering path to the same destination.
- The text projection can be replaced or removed without changing the model.

This parallels how humans use language for control: when someone says "open the cabinet," the words activate the same motor-planning representations that would be activated by seeing someone open a cabinet. Language doesn't create new representations — it indexes into existing ones.

## Text → TTS → Audio: The Simplest Control Path

Rather than building a text→latent projection at all, mediate text through audio: run TTS on the text input, feed the synthesized speech through the existing Whisper encoder. The model only ever sees audio.

Why this is appealing:

- **Single language pathway.** Whisper is the only entry point for linguistic content, whether from human speech or TTS. No text encoder, no extra projection, no contrastive alignment.
- **Pure latent space.** Shaped entirely by sensory prediction. No text-specific training signal distorting it.
- **TTS sits outside the model.** It's a preprocessing step, not architecture. Swap engines freely. The model just hears audio.
- **Grounding is automatic.** If the model trains on narrated video where real speech co-occurs with actions, synthesized speech describing the same actions should activate the same latent regions. Whisper doesn't distinguish real from synthesized speech.
- **Matches the biology.** There is no text cortex. When you read silently, the brain converts visual text to an internal auditory representation (subvocalization) and processes it through auditory pathways. Text → TTS → audio is the engineering equivalent.

The key assumption: Whisper's representations must be sufficiently invariant across real speech, accents, noise levels, and TTS output. Given that Whisper was trained on 680K+ hours of diverse speech and modern TTS is extremely naturalistic, this seems safe.

This also means the path from "text control" to "speech control" is trivial — remove the TTS step. The model already understands speech.

## Conditioned Dreaming: Video Context + Audio Control

The end-state interaction model: show the system a video (visual context), speak or TTS an instruction (audio control), and the model dreams a trajectory conditioned on both.

1. Encode seed video → latent state at all levels (where the world is now).
2. TTS instruction → Whisper → audio features (what should happen next).
3. AudioVideoEncoder fuses both streams at each prediction step.
4. Dreamer rolls forward — audio context biases predictions toward the described action.

This gives two independent control axes:
- **Video context** grounds the current world state (which kitchen, what objects are present, where things are).
- **Audio** specifies the intended trajectory (what action to take, what goal to pursue).

Same kitchen + "make coffee" → one trajectory. Same kitchen + "wash dishes" → different trajectory. Same instruction + different kitchen → kitchen-specific version of the same action.

The architecture already supports this (AudioVideoEncoder + Dreamer exist). The requirement is training data: narrated video where speech and actions co-occur, so the hierarchy learns that hearing "open the cabinet" predicts seeing a hand reach toward a cabinet.

## Conversational Video as Training Data

Video with people talking is not a special case — it's some of the richest training data available. The hierarchy processes it through the same audiovisual pathway:

- **Visual stream**: gestures, gaze, facial expressions, turn-taking body language, object references.
- **Audio stream**: both speakers through Whisper. Intonation, timing, content.
- **Prediction targets**: one person finishes speaking → predict the other starts. Question intonation → predict response. "Pass me the salt" → predict reaching motion. "Should we make dinner?" → predict movement toward kitchen.

Conversational structure (turn-taking, topic shifts, agreement/disagreement) emerges as temporal patterns in the latent space, the same way action structure emerges from non-conversational video. The model doesn't parse syntax — it learns that these sound patterns co-occur with these visual patterns and together predict these future states.

Conversational video is actually a better learning signal than narrated video for one reason: **the causal arrow from audio to vision is stronger.** In narrated cooking, speech describes what's already happening (narration follows action). In conversation, speech *causes* what happens next ("pass me the salt" → reaching motion). This direct causal link should make audio-visual correspondence easier to learn.

## Dreaming with Audio: Decoding Both Streams

The vision decoder (phase 2) maps level-1 latents back to frames. The same pattern applies to audio: an audio decoder maps level-2 latents back to mel spectrograms, which an off-the-shelf vocoder (HiFi-GAN, Vocos) converts to waveform.

Level 2 is the right source for audio — one level-2 latent covers 8 frames ≈ 250ms, which maps naturally to a short audio segment. Level 1 is too fine-grained for audio; level 3 is too coarse.

```
Dreamer: z_0 → z_1 → z_2 → ... (latent trajectory)
  ├─ VisionDecoder:  z_t^(1) → frames → video
  └─ AudioDecoder:  z_t^(2) → mel spectrogram → vocoder → waveform
```

Both decoders are trained phase-2 style with the world model frozen. They learn to read the latent space, not reshape it. The audio decoder is structurally a small transposed-conv1D or transformer that maps a level-2 latent to (n_mels, T_audio).

The result: dreamed videos with synchronized audio. A dreamed kitchen scene has both the visual trajectory and the sounds of plates, water, conversation. This is a strong signal for evaluating what the model has actually learned — if the dreamed audio matches the dreamed visuals, the latent space has captured meaningful audiovisual structure.

## Open Questions

- Does the Whisper audio encoder retain enough linguistic structure at the representation level, or does its projection to a small embedding lose the language signal?
- How much narrated video is needed for language-level concepts to emerge at level 3? Is EPIC-KITCHENS enough, or do we need Ego4D-scale data?
- Can we measure language emergence without a text encoder? (Clustering level-3 latents and checking if clusters correspond to narrated actions would be one approach.)
- Should the audio backbone be unfrozen at higher levels to let linguistic features specialize, or does the frozen Whisper representation already capture enough?
