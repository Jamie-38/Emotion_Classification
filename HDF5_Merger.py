# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:37:36 2024

@author: Jayyy
"""

import h5py
import os
import glob

def create_master_hdf5(base_directory, master_file):
    """
    Create a master HDF5 file with ExternalLink, linking to individual HDF5 files in subdirectories.
    
    Parameters:
    base_directory (str): The base directory containing subdirectories with HDF5 files.
    master_file (str): The path to the master HDF5 file to create.
    """
    all_videos = os.listdir(base_directory)
    with h5py.File(master_file, 'w') as hf_out:
        for video in all_videos:
            directory = os.path.join(base_directory, video)
            if os.path.isdir(directory):
                file_pattern = os.path.join(directory, '*.hdf5')
                files_in_dir = glob.glob(file_pattern)
                
                for file in files_in_dir:
                    video_name = os.path.splitext(os.path.basename(file))[0]
                    hf_out[video_name] = h5py.ExternalLink(file, '/')

def copy_data_to_new_hdf5(master_file, new_file):
    """
    Copy data from a master HDF5 file to a new HDF5 file.
    
    Parameters:
    master_file (str): The path to the master HDF5 file.
    new_file (str): The path to the new HDF5 file to create.
    """
    with h5py.File(master_file, 'r') as hf_master, h5py.File(new_file, 'w') as hf_new:
        for video_key in hf_master.keys():
            print(f"Processing video: {video_key}")
            video_group = hf_master[video_key]
            new_video_group = hf_new.create_group(video_key)
            
            for emotion_key in video_group.keys():
                print(f"  Processing emotion: {emotion_key}")
                emotion_group = video_group[emotion_key]
                new_emotion_group = new_video_group.create_group(emotion_key)
                
                for frame_key in emotion_group.keys():
                    print(f"    Processing frame: {frame_key}")
                    frame_group = emotion_group[frame_key]
                    new_frame_group = new_emotion_group.create_group(frame_key)
                    
                    for data_key in frame_group.keys():
                        print(f"      Found data: {data_key}")
                        data = frame_group[data_key]
                        if isinstance(data, h5py.Dataset):
                            print(f"      Copying dataset: {data_key}")
                            # Check if the dataset is scalar
                            if data.shape == ():
                                new_frame_group.create_dataset(data_key, data=data[()])
                            else:
                                new_frame_group.create_dataset(data_key, data=data[:])
                        else:
                            print(f"      Skipping non-dataset {data_key} in {frame_key}")

