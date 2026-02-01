# Emotion Classification — Multimodal Dataset Pipeline + Sequence Model (Video + Audio + Phonemes)

This repository contains a **prototype multimodal pipeline** for emotion classification from short videos, combining:

- **Facial landmarks** (per-frame, via MediaPipe face landmarker)
- **Speech audio features** (mel spectrogram segments per frame)
- **Phoneme timing** aligned to video frames (via forced alignment)

The system extracts and aligns these signals at the **frame level**, stores them in **HDF5** for flexible training workflows, then trains a **sequence model (Bi-LSTM)** over fixed-length frame windows.

> **Portfolio note:** this repo is a *case study / code snapshot*. The original dataset copy and build environment used during development are no longer available, so it is not provided as a fully reproducible training package. The focus here is the **pipeline design, data representation, and model structure**.

---

## What this demonstrates

**Applied ML / modelling**
- Multimodal feature fusion (vision + audio + text-derived timing)
- Sequence modelling for temporal classification (Bi-LSTM over frame windows)
- Practical preprocessing: padding/truncation, spectrogram normalisation, phoneme embedding

**Engineering / data pipeline**
- Frame-accurate alignment of heterogeneous signals (video timestamps ↔ audio features ↔ phoneme intervals)
- Storage of large structured training data using HDF5 (frame groups, per-sample retrieval)
- Dataset loaders feeding `tf.data` pipelines for training at scale

---

## High-level pipeline

1. **Video processing**
   - Decode video frames and compute per-frame timestamps.

2. **Face landmarks**
   - Detect dense facial landmarks per frame (e.g., face mesh).

3. **Audio features**
   - Extract mono audio and compute mel spectrogram.
   - Slice mel segments corresponding to each video frame time window.

4. **Phoneme alignment**
   - Use forced alignment to generate a TextGrid with phoneme intervals.
   - Map each frame timestamp to the phoneme active during that interval.

5. **Storage**
   - Store per-frame data (landmarks, mel segment, phoneme id) into HDF5.
   - Optional utilities to merge or link multiple HDF5 shards.

6. **Training**
   - Build fixed-length sequences (e.g., 30 frames) and pad/truncate as needed.
   - Train a multi-branch model that fuses modalities before Bi-LSTM layers.

---

## Model overview

The classifier uses three input branches per frame sequence:

- **Landmarks branch:** flatten + dense layers over face landmark coordinates  
- **Mel branch:** small CNN over per-frame mel segments  
- **Phoneme branch:** embedding of per-frame phoneme ids  

These are concatenated and fed into **Bidirectional LSTMs** for temporal modelling, ending in an 8-class softmax output (RAVDESS emotion set).

---

## Repo structure

- `Video_Controller.py` — reads video frames and timestamps  
- `Audio_Controller.py` — extracts audio (ffmpeg) and generates mel spectrogram segments  
- `Face_Landmark_Generator.py` — runs MediaPipe face landmark detection  
- `Aligner.py` + `run_mfa.bat` — runs Montreal Forced Aligner (MFA) to generate TextGrid alignments  
- `TextGrid_Controller.py` — parses TextGrid and exposes phoneme intervals  
- `Training_Frame.py` — frame-level container (landmarks, mel segment, phoneme)  
- `Storage_Controller.py` — writes per-frame data to HDF5  
- `HDF5_Merger.py` — links/merges multiple HDF5 files  
- `Emotion_Classifier.py` — model definition (multimodal + Bi-LSTM)  
- `Interface.py` — end-to-end dataset processing prototype  
- `Interface_Model.py` — dataset loading (`tf.data`) + training loop prototype

---

## Notes / limitations

- Hard-coded local paths and environment-specific scripts reflect the original development setup.
- This snapshot is intended to showcase approach and structure rather than provide a turnkey training environment.
- The design is suitable for extension into related tasks such as:
  - viseme prediction / lip-sync conditioning
  - multimodal affect modelling
  - real-time inference with streaming feature extraction

---

## Context

This project started as a hobby R&D build to explore **frame-level multimodal alignment** and **sequence modelling** for emotion classification using standard research datasets and tooling.
