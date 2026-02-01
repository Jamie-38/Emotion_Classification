# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 18:35:55 2024

@author: Jayyy
"""
import h5py

class HDF5_Container:
    """
    A class to handle HDF5 file operations for video data.
    """
    def __init__(self, path):
        """
        Initialize an HDF5_Container instance.
        
        Parameters:
        path (str): The path to the HDF5 file.
        """        
        self.path = path

    def read_video_data(self):
        """
        Generator function to read video data from the HDF5 file.
        
        Yields:
        dict: A dictionary containing video name, emotion, and frames with landmarks, mel spectrogram, and phoneme.
        """
        with h5py.File(self.path, 'r') as file:
            video_names = list(file.keys())
            for video_name in video_names:
                video_group = file[video_name]
                for emotion in video_group.keys():
                    emotion_group = video_group[emotion]
                    emotion_data = {}
                    for frame_index in sorted(emotion_group, key=lambda x: int(x)):
                        frame_group = emotion_group[frame_index]

                        landmarks = frame_group['landmarks'][:]
                        mel = frame_group['mel'][:]
                        phoneme = frame_group['phoneme'][()]

                        emotion_data[int(frame_index)] = {
                            'landmarks': landmarks,
                            'mel': mel,
                            'phoneme': phoneme
                        }

                    all_data = {
                        'video_name': video_name,
                        'emotion': emotion,
                        'frames': emotion_data
                    }

                    yield all_data
