# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:46:03 2024

This script defines a Bi-LSTM model for emotion classification based on landmarks, mel spectrogram, and phoneme data.

@author: Jayyy
"""
from tensorflow.keras import layers, models, Input



# Define input shapes
sequence_length = 30  # Number of frames in each sequence
num_landmarks = 478
num_mels = 128
mel_length = 64
num_phonemes = 91  # Number of unique phonemes
num_emotions = 8  # Number of emotion classes

def create_emotion_classifier():
    """
    Create a Bi-LSTM model for emotion classification.
    
    Returns:
    model (tf.keras.Model): The compiled Keras model.
    """
    # Landmarks branch
    landmark_input = Input(shape=(sequence_length, num_landmarks, 3), name='landmarks')
    x_landmark = layers.TimeDistributed(layers.Flatten())(landmark_input)
    x_landmark = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x_landmark)
    x_landmark = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x_landmark)

    # Mel spectrograms branch
    mel_input = Input(shape=(sequence_length, num_mels, mel_length, 1), name='mel_spectrogram')  # Add a channel dimension
    x_mel = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'))(mel_input)
    x_mel = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x_mel)
    x_mel = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu'))(x_mel)
    x_mel = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x_mel)
    x_mel = layers.TimeDistributed(layers.Flatten())(x_mel)
    x_mel = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x_mel)

    # Phonemes branch
    phoneme_input = Input(shape=(sequence_length, 1), name='phonemes')
    x_phoneme = layers.TimeDistributed(layers.Embedding(input_dim=num_phonemes, output_dim=64))(phoneme_input)
    x_phoneme = layers.TimeDistributed(layers.Flatten())(x_phoneme)
    x_phoneme = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x_phoneme)

    # Concatenate branches
    x = layers.Concatenate()([x_landmark, x_mel, x_phoneme])

    # Bi-LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)

    # Dense output layer
    output = layers.Dense(num_emotions, activation='softmax', name='emotion_output')(x)

    # Define model
    model = models.Model(inputs=[landmark_input, mel_input, phoneme_input], outputs=output)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    model = create_emotion_classifier()
    model.summary()
