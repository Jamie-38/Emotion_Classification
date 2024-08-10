# Emotion_Classification

This Project is an attempt to build a bi-lstm model for emotion classification through analysis of facial landmarks and mel spectrogram data of speech.

The project makes use of the RAVDESS dataset. This dataset is made up of a series of videos by a number of actors in which one of two lines is spoken while expressing one of eight emotions. From these videos, the project gathers facial landmarks, mel spectrograms and phonemes.

For each video, this data is gathered for each frame and subsequently stored in HDF5. This is to allow greater control over training data. The dataset contains both male and female actors, with 60 videos per actor. By processing and storing the data for each video individually, the project can be more selective on which data is used. For example, the dataset contains both speaking and singing. Initially, this project only made use of speaking videos.

In order to align phonemes correctly with the corresponding video frame, Montreal Forced Aligner is used.

The project has been built in three sections to separate concerns, the first deals with processing the video dataset to extract the relevant data and store it. The second aligns the phonemes with the frame data, completing the training data. Finally, the third section handles the manipulation of the training data, and building and training a model.
