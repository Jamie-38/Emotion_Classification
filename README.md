# Emotion_Classification

This Project is an attempt to build a bi-lstm model for emotion classification through analysis of facial landmarks and melspectogram data of speech.

The project begins by processing the RAVDESS dataset. This dataset is made up of a series of videos by a number of actors in which one of two lines are spoken while expressing one of eight emotions. From these videos the projet gathers facial landmarks, mel spectograms and phonemes.

For each video, this data is gathered for each frame and subsiquently stored in HDF5. This is to allow greater control over training data. The dataset contains both male and female actors.
