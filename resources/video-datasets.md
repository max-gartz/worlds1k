# Video Datasets for Hierarchical Predictive World Models

Curated list of Hugging Face datasets for training models that predict next visual states from video sequences. Ranked by practical usefulness for prototyping.

---

## Tier 1: Best for Getting Started

These datasets are most directly useful — they either have explicit action signals, ego-motion, or were literally built for world modeling.

### 1X World Model Challenge Dataset ⭐
- **HF Path:** [`1x-technologies/worldmodel`](https://huggingface.co/datasets/1x-technologies/worldmodel)
- **Size:** VQ-encoded video patches (16×16 at 30Hz), 256×256 RGB decodable
- **What's in it:** Vector-quantized video with aligned action arrays (joint positions, driving commands, neck orientation). Includes segment IDs for episode boundaries.
- **Why it's relevant:** Purpose-built for world model training. Already tokenized into discrete patches — you can train a next-token predictor directly. Actions are aligned with frames, giving you the action-conditioned prediction setup out of the box.
- **Also see:** [`1x-technologies/world_model_tokenized_data`](https://huggingface.co/datasets/1x-technologies/world_model_tokenized_data) (compression challenge) and [`1x-technologies/world_model_raw_data`](https://huggingface.co/datasets/1x-technologies/world_model_raw_data) (512×512 raw video)

### Ego4D
- **HF Path:** [`wofmanaf/ego4d-video`](https://huggingface.co/datasets/wofmanaf/ego4d-video)
- **Size:** ~3,670 hours of egocentric video
- **What's in it:** 931 camera wearers across 74 locations in 9 countries. Multi-modal: video, 3D scans, audio, gaze, stereo, narrations. Daily activities, object interactions, social scenes.
- **Why it's relevant:** Massive first-person video where camera motion = implicit action signal. The continuous nature of daily activities gives you natural next-state prediction targets. Multi-modal signals (gaze, narration) can serve as auxiliary supervision for hierarchical representations.

### EPIC-KITCHENS-100
- **HF Path:** [`awsaf49/epic_kitchens_100`](https://huggingface.co/datasets/awsaf49/epic_kitchens_100) (also `HuggingFaceM4/epic_kitchens_100`)
- **Size:** 100 hours unscripted footage, 55 hours with 11.5M frames, 39.6K action segments
- **What's in it:** First-person kitchen video from 32 participants in their own kitchens. Dense hand-object interaction annotations, 454K bounding boxes.
- **Why it's relevant:** Unscripted continuous egocentric video with rich manipulation sequences. Object state changes (cutting, pouring, opening) make it ideal for learning causal next-state prediction. Relatively manageable size for prototyping.

### NVIDIA PhysicalAI Autonomous Vehicles
- **HF Path:** [`nvidia/PhysicalAI-Autonomous-Vehicles`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- **Size:** 1,700 hours of driving video; 306K multi-camera clips
- **What's in it:** Driving data from 25 countries, 2,500+ cities. Multi-camera, LiDAR (298K clips), radar (160K clips). Organized by geography and weather conditions.
- **Why it's relevant:** Strong ego-motion signal from vehicle movement. Multi-sensor data enables cross-modal world modeling. Diverse conditions (weather, time, geography) force the model to learn robust scene dynamics.

---

## Tier 2: Strong Candidates

Good datasets with clear ego-motion or continuous scene dynamics, but may need more preprocessing or have less direct action signals.

### BDD100K
- **HF Path:** [`dgural/bdd100k`](https://huggingface.co/datasets/dgural/bdd100k)
- **Size:** 100,000 videos (~40 seconds each)
- **What's in it:** Dashboard camera driving video with object detection, weather, time-of-day, and scene labels. Diverse driving conditions across the US.
- **Why it's relevant:** Massive scale with continuous forward-facing driving video. 40-second clips are long enough for temporal prediction. Rich condition annotations enable conditional generation experiments.

### OpenDV-YouTube-Language
- **HF Path:** [`OpenDriveLab/OpenDV-YouTube-Language`](https://huggingface.co/datasets/OpenDriveLab/OpenDV-YouTube-Language)
- **Size:** YouTube driving clips with command annotations
- **What's in it:** Autonomous driving video from YouTube with driving command annotations and BLIP-generated descriptions per frame.
- **Why it's relevant:** Specifically designed for driving world models. Command annotations give you action-conditioned prediction. Used in benchmarks against nuScenes and Waymo.

### MultiCamVideo-Dataset
- **HF Path:** [`KlingTeam/MultiCamVideo-Dataset`](https://huggingface.co/datasets/KlingTeam/MultiCamVideo-Dataset) (also `KwaiVGI/MultiCamVideo-Dataset`)
- **Size:** 136K videos from 13.6K scenes; 112K unique camera trajectories
- **What's in it:** Synthetic multi-camera video rendered in Unreal Engine 5. Indoor/outdoor scenes: city streets, malls, cafes, offices, countryside. 10 cameras per scene with diverse movements.
- **Why it's relevant:** Known camera trajectories = perfect action labels. Multi-view coverage of same scenes enables learning 3D-consistent representations. Synthetic data means clean, controllable training signal.

### Something-Something v2
- **HF Path:** [`HuggingFaceM4/something_something_v2`](https://huggingface.co/datasets/HuggingFaceM4/something_something_v2)
- **Size:** 220,847 labeled video clips
- **What's in it:** Fine-grained human hand-object interactions: putting things in containers, turning objects upside down, covering, pushing, pulling. Pre-defined action templates with everyday objects.
- **Why it's relevant:** Each clip shows a clear state transition caused by an action. The templated structure (action + object → new state) is exactly the causal structure a world model should learn. Good for validating that your model captures physical causality.

### NVIDIA Cosmos Drive Dreams
- **HF Path:** [`nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams)
- **Size:** 5,843 real 10-second clips + 81,802 synthetic 121-frame videos
- **What's in it:** Real + synthetically generated driving video with challenging scenarios (rain, snow, fog). Semantic labels included.
- **Why it's relevant:** Synthetic augmentation of rare scenarios. Real+synthetic mix is useful for training robust next-frame prediction in conditions underrepresented in real data.

---

## Tier 3: Useful for Scale or Specific Angles

These add breadth — more diverse scenes, more data, or specific domain coverage.

### FineVideo
- **HF Path:** [`HuggingFaceFV/finevideo`](https://huggingface.co/datasets/HuggingFaceFV/finevideo)
- **Size:** 43,751 YouTube videos (~3,425 hours); avg 4.7 min per video
- **What's in it:** Videos across 122 categories with rich annotations: scene descriptions, character interactions, plot analysis, mood, audio-visual harmony.
- **Why it's relevant:** Long-form continuous video (not short clips). Good for training hierarchical temporal models that need to maintain coherence over minutes, not seconds. The narrative structure maps well to hierarchical state prediction.

### Kinetics-400 / Kinetics-700
- **HF Path:** [`liuhuanjim013/kinetics400`](https://huggingface.co/datasets/liuhuanjim013/kinetics400), [`atalaydenknalbant/Kinetics-700`](https://huggingface.co/datasets/atalaydenknalbant/Kinetics-700)
- **Size:** 400 classes / 700 classes, ~10 second clips each from YouTube
- **What's in it:** Human action recognition benchmark. Realistic YouTube videos of human-object and human-human interactions across hundreds of activity types.
- **Why it's relevant:** Massive diversity of visual dynamics. Pre-training on this teaches the model broad priors about how the visual world changes. Short clips are a limitation but useful for learning local dynamics.

### ShareGPT4Video
- **HF Path:** [`ShareGPT4Video/ShareGPT4Video`](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video)
- **Size:** 4.8M videos with captions + 40K with detailed GPT-4V captions
- **What's in it:** YouTube + user-uploaded videos with detailed scene understanding captions covering spatial relationships, object properties, world knowledge, aesthetics.
- **Why it's relevant:** The detailed captions provide semantic grounding for visual state transitions. Could be useful for training a language-conditioned world model or for providing hierarchical supervision at the semantic level.

### NVIDIA PhysicalAI SmartSpaces
- **HF Path:** [`nvidia/PhysicalAI-SmartSpaces`](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)
- **Size:** 250+ hours from ~1,500 cameras
- **What's in it:** Indoor video from warehouses, hospitals, retail environments. Multi-camera coverage of interior spaces.
- **Why it's relevant:** Real indoor scene dynamics with people, objects, and activities. Fixed + moving camera perspectives for indoor world modeling.

### InternScenes
- **HF Path:** [`InternRobotics/InternScenes`](https://huggingface.co/datasets/InternRobotics/InternScenes)
- **Size:** ~40,000 scenes with 1.96M 3D objects
- **What's in it:** Large-scale interactive indoor scene dataset with realistic layouts. 15 scene types, 288 object classes. 20% of objects are interactive (cabinets, microwaves, drawers).
- **Why it's relevant:** Interactive objects with known state transitions. If you can render trajectories through these scenes, you get perfect ground truth for next-state prediction in manipulable environments.

### Egocentric-10K
- **HF Path:** [`builddotai/Egocentric-10K`](https://huggingface.co/datasets/builddotai/Egocentric-10K)
- **Size:** 10K egocentric video samples
- **What's in it:** First-person video from real factories. High hand visibility and dense manipulation sequences.
- **Why it's relevant:** Industrial manipulation domain. Niche but valuable if you want to test world model generalization to factory settings.

---

## Tier 4: Robotics-Specific (Action-Conditioned)

These have explicit robot actions paired with video — ideal if your world model is targeting embodied AI.

### RoboMIND
- **HF Path:** [`x-humanoid-robomind/RoboMIND`](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)
- **Size:** 107K demonstration trajectories, 479 tasks
- **What's in it:** Multi-embodiment teleoperation data from Franka Panda (52.9K), humanoid (19.2K), dual-arm (10.6K), UR-5e (25.2K). Multi-view observations, proprioceptive state, language task descriptions. 5K failure demonstrations.
- **Why it's relevant:** Largest multi-embodiment manipulation dataset. Explicit action → next-state pairs across different robots. Failure cases are gold for learning what doesn't happen.

### NVIDIA PhysicalAI Robotics Manipulation
- **HF Path:** [`nvidia/PhysicalAI-Robotics-Manipulation-SingleArm`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-SingleArm), [`nvidia/PhysicalAI-Robotics-Manipulation-Augmented`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Augmented)
- **Size:** Multiple sub-datasets in LeRobot format
- **What's in it:** Franka Panda manipulation videos (stacking, cabinet opening, drawer opening). World + hand cameras, depth, segmentation, surface normals, full robot/object state.
- **Why it's relevant:** Multi-modal manipulation data with semantic segmentation. The combination of RGB + depth + segmentation enables learning structured state representations.

---

## Quick-Start Recommendation

For a prototype hierarchical predictive world model, start with:

1. **1X World Model dataset** — already tokenized, has actions, purpose-built for this
2. **EPIC-KITCHENS-100** — manageable size, continuous ego-video, clear state transitions
3. **BDD100K** — scale + continuous driving ego-motion for outdoor scenes

Then scale up with Ego4D and PhysicalAI-Autonomous-Vehicles once the architecture works.

---

*Last updated: March 2026*
