# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:52:22 2024

@author: Jayyy
"""
from video_controller import VideoController
from FaceLandmarkGenerator import FaceLandMarkGenerator
from AudioController import AudioController
from Aligner import run_mfa_alignment
from StorageController import HDF5_Container
from TrainingFrame import Training_Frame
from TextGridController import Read_Textgrid

import os
from pathlib import Path
import h5py
import numpy as np

np.set_printoptions(precision=17)

landmark_model_path = 'E:/projects/face/spyder_project/face/face_landmarker.task'
actor_directory = 'E:/projects/face/media/unziped/Actor_03/'

# MFA
model_directory = 'E:/projects/face/MFA/pretrained_models/acoustic/english_mfa.zip'
dictionary_path = 'E:/projects/face/MFA/pretrained_models/dictionary/english_mfa.dict'
output_path = "E:/projects/face/MFA/output/"

def split_file_name(file_name):
    """
    Split the file name into its components.
    
    Parameters:
    file_name (str): The file name to split.
    
    Returns:
    list: A list of components of the file name.
    """
    return file_name.split("-")

def create_statement_txt(statement_id, statement_path):
    """
    Create a text file with the statement corresponding to the statement ID.
    
    Parameters:
    statement_id (str): The ID of the statement.
    statement_path (str): The path to save the text file.
    """
    kids_statement = "kids are talking by the door"
    dogs_statement = "dogs are sitting by the door"
    
    if statement_id == "01":
        statement = kids_statement
    elif statement_id == "02":
        statement = dogs_statement
    else:
        raise ValueError(f"Unknown statement ID: {statement_id}")
        
    with open(statement_path, "w") as txt_file:
        txt_file.write(statement)
        
def landmarks_to_txt(output_path, timestamp, landmarks):
    """
    Save the landmarks to a text file.
    
    Parameters:
    output_path (str): The path to save the text file.
    timestamp (int): The timestamp of the frame.
    landmarks (list): The facial landmarks.
    """
    
    landmark_str = "TIMESTAMP: " + str(timestamp) + " " +  "LANDMARKS: " + str(landmarks)
    
    with open(output_path, "a") as txt_file:
        txt_file.write(landmark_str)
        
def retrive_phoneme(timestamp, phones):
    """
    Retrieve the phoneme at a given timestamp.
    
    Parameters:
    timestamp (int): The timestamp to retrieve the phoneme for.
    phones (list): The list of phonemes with their timing information.
    
    Returns:
    str: The phoneme label.
    """
    for phone in phones:
        phone_start_ms = int(phone.xmin * 1000)
        phone_end_ms = int(phone.xmax * 1000)
        if timestamp in range(phone_start_ms, phone_end_ms):
            label = phone.text.transcode()
            return label

def combine_mel_segments(training_frames):
    """
    Concatenate all mel spectrogram segments to reconstruct the full spectrogram.
    
    Parameters:
    training_frames (list): The list of Training_Frame objects.
    
    Returns:
    np.ndarray: The full mel spectrogram.
    """    
    full_mel_spectrogram = np.concatenate([frame.mel_segment for frame in training_frames], axis=1)
    
    return full_mel_spectrogram

def combine_mel_segments_HDF5(all_data):
    """
    Concatenate all mel spectrogram segments from HDF5 data to reconstruct the full spectrogram.
    
    Parameters:
    all_data (dict): The data read from the HDF5 file.
    
    Returns:
    np.ndarray: The full mel spectrogram.
    """    
    read_frames = all_data['frames']
    full_mel_spectrogram = np.concatenate([frame_data['mel'] for frame_index, frame_data in read_frames.items()], axis=1)
    
    return full_mel_spectrogram

def print_training_frames(training_frames):
    """
    Print the details of each training frame.
    
    Parameters:
    training_frames (list): The list of Training_Frame objects.
    """
    for frame in training_frames:
        print("frame_index: ", frame.frame_index)
        print("Phone: ", frame.phoneme)
        print("LandMarks: ", frame.landmarks)


# Process each video in the actor directory   
for video in os.listdir(actor_directory): 
    
    # Get emotion ID and spoken statement from video name
    file_name = Path(video).stem
    filename_ids = split_file_name(file_name)
    statement_id = filename_ids[4]
    emotion_id = filename_ids[2]
    
    video_path = os.path.join(actor_directory, video)   
    
    # Only process videos with audio
    if filename_ids[0] == "01":
        
        video_path = os.path.join(actor_directory, video)
        
        if os.path.isfile(video_path):
            print(video_path)

        output_dir = os.path.join(output_path, file_name)
        os.makedirs(output_dir, exist_ok=True)
    
        converted_audio_output_path = os.path.join(output_dir, file_name + '.wav')
        statement_path = os.path.join(output_dir, file_name + '.txt')
        textgrid_path = os.path.join(output_dir, file_name + '.TextGrid')        
        HDF5_file_path = os.path.join(output_dir, file_name + '.hdf5')
        
        landmark_gen = FaceLandMarkGenerator(landmark_model_path)  
        video_controller = VideoController(video_path)
        audio_controller = AudioController(video_path, converted_audio_output_path)       
        
        create_statement_txt(statement_id, statement_path)
        run_mfa_alignment(output_dir, model_directory, dictionary_path, output_dir)
        textgrid = Read_Textgrid(textgrid_path)
        phones = textgrid.grid['phones']       
        
        training_frames = []
    
        # Process video frames
        for frame, timestamp, frame_index in video_controller.process_video():            
            
            face_landmarks_list = landmark_gen.find_landmarks(frame, timestamp)
            phoneme = retrive_phoneme(timestamp, phones)
            mel_segment = audio_controller.retrive_mel_segment(timestamp, video_controller.frame_duration_ms)
            
            training_frame = Training_Frame(statement_id, frame_index, face_landmarks_list, phoneme, mel_segment) 
            training_frames.append(training_frame)
            
            
            landmark_gen.draw_landmarks(frame, face_landmarks_list)        
            #video_controller.show_frame(frame)
    
            
        # Create an instance of HDF5_Container and add data
        hdf5_container = HDF5_Container(HDF5_file_path)
        hdf5_container.create_hdf5_file()
        
        # Add data to HDF5
        hdf5_container.add_video_data_batch(file_name, emotion_id, training_frames)
        hdf5_container.close_hdf5_file()
        
        # Read all data from the HDF5 file
        all_data = hdf5_container.read_video_data(HDF5_file_path)
        
        # Access and process the data
        #read_video_name = all_data['video_name']
        read_emotion = all_data['emotion']
        read_frames = all_data['frames']
        
        full_mel = combine_mel_segments_HDF5(all_data)
        audio_controller.show_melspectrogram(full_mel, audio_controller.sr, audio_controller.hop_length)
        
        
    
        

