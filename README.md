# Emotion Classification — Frame-Level Multimodal Dataset Pipeline + Sequence Model (Video + Audio + Phonemes)

This repository contains a **prototype multimodal ML pipeline** for emotion classification from short videos by aligning three signals at the **video-frame level**:

- **Facial landmarks** per frame (MediaPipe face landmarker)
- **Speech audio features** sliced per frame (mel spectrogram segments)
- **Phoneme timing** mapped to frames via forced alignment (Montreal Forced Aligner → TextGrid)

The core output is a structured **HDF5 dataset** designed for scalable training and flexible retrieval, plus a reference **multimodal Bi-LSTM** model that fuses the modalities over fixed-length frame windows.

> **Portfolio note:** this repo is a *case study / code snapshot*. The original dataset copy and exact build environment used during development are no longer available, so it is not provided as a turnkey, fully reproducible training package. The focus here is the **pipeline design, data representation, and model structure**.

---

## Why this project exists

A lot of “multimodal emotion recognition” work collapses into “throw features into a model.” The hard part in practice is getting **heterogeneous signals aligned reliably**:

- video frames have timestamps and variable FPS
- audio features operate on hop lengths / windows
- phonemes exist as continuous time intervals
- training wants fixed-length sequences and consistent shapes

This project explores that real-world friction and builds a pipeline that produces **frame-aligned training examples** suitable for sequence modelling.

---

## What the system does

1. **Decode video frames + timestamps**  
   Read frames and compute per-frame timestamps based on FPS.

2. **Extract face landmarks per frame**  
   Run MediaPipe face landmark detection on each frame (dense landmark set).

3. **Extract audio + mel spectrogram**  
   Use FFmpeg to extract mono audio and compute a mel spectrogram with librosa.

4. **Forced alignment → phoneme intervals**  
   Run Montreal Forced Aligner (MFA) to create a TextGrid of phoneme timings.

5. **Map phonemes to frames**  
   For each video frame timestamp, select the phoneme interval active at that time.

6. **Store frame-level training data in HDF5**  
   Write per-frame data (landmarks, mel slice, phoneme id) into a structured HDF5 layout.

7. **Train a multimodal sequence model**  
   Load sequences (pad/truncate to fixed length), fuse modalities, and train a Bi-LSTM classifier.

---

## Data representation (HDF5 schema)

Data is stored as:
/<video_name>/<emotion>/<frame_index>/{landmarks, mel, phoneme}


- `landmarks`: `[num_landmarks, 3]` (x, y, z per landmark)
- `mel`: `[n_mels, time_frames]` (mel slice aligned to the video frame window)
- `phoneme`: integer ID mapped from a phoneme label
- `<emotion>` uses a zero-padded string (e.g., `"01"`, `"02"`, ...)

This structure is intentionally “inspectable” with tools like `h5py`, making debugging alignment and preprocessing much easier than opaque binary blobs.

---

## Model overview

The reference classifier is a three-branch multimodal network operating on fixed-length sequences:

- **Landmarks branch:** flatten + dense layers over face landmark coordinates per frame  
- **Mel branch:** small CNN over per-frame mel slices  
- **Phoneme branch:** embedding of per-frame phoneme IDs  

The branches are concatenated per timestep and passed through **Bidirectional LSTMs** for temporal modelling, with an 8-class softmax output (RAVDESS emotion set).

> The model is deliberately a “strong baseline” rather than a cutting-edge architecture — the main focus is the dataset pipeline and alignment.

---

## Repo structure

- `interface.py` — end-to-end dataset processing pipeline (video → landmarks/audio/phonemes → HDF5)
- `video_controller.py` — reads video frames + computes per-frame timestamps
- `face_landmark_generator.py` — MediaPipe face landmark extraction
- `audio_controller.py` — FFmpeg audio extraction + mel spectrogram + per-frame mel slicing
- `aligner.py` + `run_mfa.bat` — MFA runner (forced alignment) producing TextGrid files
- `textgrid_controller.py` — parses TextGrid phoneme intervals
- `training_frame.py` — frame-level container (emotion, frame index, landmarks, phoneme, mel slice)
- `storage_controller.py` — writes aligned per-frame data into HDF5
- `hdf5_merger.py` — utilities for linking/merging multiple HDF5 shards
- `interface_model.py` — HDF5 loader + `tf.data` pipeline + training loop
- `emotion_classifier.py` — multimodal Bi-LSTM model definition

---

## What this demonstrates (portfolio framing)

**ML / modelling**
- Multimodal feature fusion (vision + audio + text-derived timing)
- Sequence modelling (Bi-LSTM over frame windows)
- Practical preprocessing (padding/truncation, mel normalisation, phoneme embedding)

**Data / engineering**
- Frame-accurate alignment of heterogeneous signals (timestamps ↔ hop windows ↔ phoneme intervals)
- Structured storage of large training corpora using HDF5
- Dataset loaders feeding `tf.data` pipelines suitable for scaling

---

## Notes / limitations

- This snapshot reflects an R&D setup and includes environment-specific elements (e.g., MFA invocation, original local paths in the development machine).
- It is intended to showcase **approach and structure**, not provide a turnkey training environment.
- A productionised version would typically add:
  - pinned dependencies + containerised setup
  - clear dataset acquisition scripts
  - automated evaluation metrics + experiment tracking
  - more robust phoneme vocabulary handling and alignment QA checks

---

## Context

This started as a hobby R&D project to explore **frame-level multimodal alignment** and **sequence modelling** for emotion classification using standard research tooling and datasets (e.g., RAVDESS + Montreal Forced Aligner).
