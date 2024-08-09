# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:10:24 2024

@author: Jayyy
"""


class Training_Frame:
    """
    A class to represent a training frame with emotion label, frame index,
    facial landmarks, phoneme, and mel spectrogram segment.
    """
    
    def __init__(self, emo_label, frame_index, landmarks, phoneme, mel_segment):
        
        """
        Initialize a Training_Frame instance.
        
        Parameters:
        emo_label (str): The emotion label for the frame.
        frame_index (int): The index of the frame.
        landmarks (list): The facial landmarks for the frame.
        phoneme (str): The phoneme associated with the frame.
        mel_segment (np.ndarray): The mel spectrogram segment for the frame.
        """
        
        self.emo_label = emo_label
        self.frame_index = frame_index
        self.landmarks = landmarks
        self.phoneme = phoneme
        self.mel_segment = mel_segment

        