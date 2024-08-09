import numpy as np
import random
import tensorflow as tf
import datetime
import h5py
import time
from Storage_Controller import HDF5_Container
from Emotion_Classifier import create_emotion_classifier

# Define constants
sequence_length = 30
num_landmarks = 478
num_mels = 128
mel_target_time_frames = 64
num_phonemes = 91
num_emotions = 8

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def pad_mel_segment(mel_segment, target_length):
    """
    Pad or truncate the mel spectrogram segment to the target length.
    
    Parameters:
    mel_segment (np.ndarray): The input mel spectrogram segment.
    target_length (int): The target length for padding or truncation.
    
    Returns:
    np.ndarray: The padded or truncated mel spectrogram segment.
    """
    current_length = mel_segment.shape[1]
    if current_length < target_length:
        padding_amount = target_length - current_length
        padded_segment = np.pad(mel_segment, ((0, 0), (0, padding_amount)), 'constant')
    else:
        padded_segment = mel_segment[:, :target_length]
    return padded_segment

def normalize_mel_spectrogram(mel):
    """
    Normalize the mel spectrogram to have zero mean and unit variance.
    
    Parameters:
    mel (np.ndarray): The input mel spectrogram.
    
    Returns:
    np.ndarray: The normalized mel spectrogram.
    """
    mel_mean = np.mean(mel)
    mel_std = np.std(mel)
    if mel_std == 0:
        return np.zeros_like(mel)
    return (mel - mel_mean) / mel_std

def pad_or_truncate_sequence(sequence, target_length):
    """
    Pad or truncate the sequence to the target length.
    
    Parameters:
    sequence (np.ndarray): The input sequence.
    target_length (int): The target length for padding or truncation.
    
    Returns:
    np.ndarray: The padded or truncated sequence.
    """
    current_length = len(sequence)
    if current_length < target_length:
        padding = np.zeros((target_length - current_length, *sequence[0].shape), dtype=sequence[0].dtype)
        return np.concatenate([sequence, padding], axis=0)
    return sequence[:target_length]

def split_metadata(metadata, test_size=0.2):
    """
    Split metadata into training and testing sets.
    
    Parameters:
    metadata (list): The metadata to split.
    test_size (float): The proportion of the dataset to include in the test split.
    
    Returns:
    tuple: Training and testing metadata.
    """
    random.shuffle(metadata)
    split_index = int(len(metadata) * (1 - test_size))
    train_metadata = metadata[:split_index]
    test_metadata = metadata[split_index:]
    return train_metadata, test_metadata

class HDF5Dataset:
    """
    A class to handle dataset loading from HDF5 files.
    """
    def __init__(self, hdf5_path):
        """
        Initialize an HDF5Dataset instance.
        
        Parameters:
        hdf5_path (str): The path to the HDF5 file.
        """
        self.hdf5_path = hdf5_path

    def __call__(self, video_name, emotion):
        """
        Load data for a specific video and emotion from the HDF5 file.
        
        Parameters:
        video_name (str): The name of the video.
        emotion (int): The emotion label.
        
        Returns:
        tuple: A tuple containing the input data and the emotion label.
        """
        start_time = time.time()
        with h5py.File(self.hdf5_path, 'r') as file:
            try:
                video_group = file[video_name]
                emotion_str = f"{int(emotion):02d}"
                if emotion_str not in video_group:
                    raise KeyError(f"Emotion {emotion_str} not found for video {video_name}")

                emotion_group = video_group[emotion_str]
                landmarks = []
                mels = []
                phonemes = []

                for frame_key in sorted(emotion_group.keys(), key=int):
                    frame_group = emotion_group[frame_key]
                    landmarks.append(frame_group['landmarks'][:])
                    mel = frame_group['mel'][:]
                    mel = pad_mel_segment(mel, mel_target_time_frames)
                    mel = normalize_mel_spectrogram(mel)
                    mel = np.expand_dims(mel, axis=-1)
                    mels.append(mel)
                    phonemes.append(frame_group['phoneme'][()])

                landmarks = pad_or_truncate_sequence(np.array(landmarks), sequence_length)
                mels = pad_or_truncate_sequence(np.array(mels), sequence_length)
                phonemes = pad_or_truncate_sequence(np.array(phonemes), sequence_length)
                phonemes = np.expand_dims(phonemes, axis=-1)

                load_time = time.time() - start_time
                print(f"Time taken to load {video_name}: {load_time:.6f} seconds")

                return (landmarks, mels, phonemes), int(emotion) - 1

            except KeyError as e:
                print(f"KeyError: {e}")
                raise

def create_tf_dataset(metadata, batch_size, hdf5_path):
    """
    Create a TensorFlow dataset from metadata and HDF5 data.
    
    Parameters:
    metadata (list): The metadata for the dataset.
    batch_size (int): The batch size for training.
    hdf5_path (str): The path to the HDF5 file.
    
    Returns:
    tf.data.Dataset: The TensorFlow dataset.
    """
    hdf5_dataset = HDF5Dataset(hdf5_path)

    def generator():
        for video_name, emotion in metadata:
            yield hdf5_dataset(video_name, emotion)

    output_signature = (
        (
            tf.TensorSpec(shape=(sequence_length, num_landmarks, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(sequence_length, num_mels, mel_target_time_frames, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(sequence_length, 1), dtype=tf.int32)
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.shuffle(buffer_size=len(metadata))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: ((tf.convert_to_tensor(x[0]), tf.convert_to_tensor(x[1]), tf.convert_to_tensor(x[2])), tf.one_hot(y, num_emotions)), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.repeat()

    return dataset

if __name__ == "__main__":

    HDF5_file_path = 'E:/projects/face_model/training_data/merged_data_file.hdf5'

    data = HDF5_Container(HDF5_file_path)
    metadata = [(video_data['video_name'], video_data['emotion']) for video_data in data.read_video_data()]

    train_metadata, test_metadata = split_metadata(metadata)

    batch_size = 8  # Adjusted batch size for better utilization
    train_dataset = create_tf_dataset(train_metadata, batch_size, HDF5_file_path)
    test_dataset = create_tf_dataset(test_metadata, batch_size, HDF5_file_path)

    train_steps_per_epoch = len(train_metadata) // batch_size
    test_steps_per_epoch = len(test_metadata) // batch_size

    model = create_emotion_classifier()

    logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)

    for epoch in range(1000):
        print(f"Epoch {epoch + 1}/{1000}")
        epoch_start_time = time.time()
        model.fit(train_dataset, epochs=1, steps_per_epoch=train_steps_per_epoch, validation_data=test_dataset, validation_steps=test_steps_per_epoch, callbacks=[tensorboard_callback])
        print(f"Time taken for epoch {epoch + 1}: {time.time() - epoch_start_time:.2f} seconds")
