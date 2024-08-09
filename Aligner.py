# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:43:19 2024

@author: Jayyy
"""

import subprocess
import os

def run_mfa_alignment(input_path, model_directory, dictionary_path, output_directory):
    """
    Run MFA alignment using a subprocess call to the MFA virtual environment.

    :param audio_file_path: Path to the audio file.
    :param text_file_path: Path to the text file.
    :param model_directory: Path to the directory containing the MFA acoustic model.
    :param dictionary_path: Path to the pronunciation dictionary file.
    :param output_directory: Path to the directory where the output will be saved.
    :return: None
    """
    
    if not input_path.endswith('/'):
        input_path += '/'
    if not output_directory.endswith('/'):
        output_directory += '/'
    
    
    # Path to the batch script
    batch_script_path = 'E:/projects/face/spyder_project/face/run_mfa.bat'

    command = [
        batch_script_path,
        input_path,        
        dictionary_path,
        model_directory,
        output_directory
    ]

    # Log the command and paths for debugging
    print(f"Running command: {' '.join(command)}")
    print(f"input  path: {input_path} - Exists: {os.path.exists(input_path)}")    
    print(f"Model directory: {model_directory} - Exists: {os.path.exists(model_directory)}")
    print(f"Dictionary path: {dictionary_path} - Exists: {os.path.exists(dictionary_path)}")
    print(f"Output directory: {output_directory} - Exists: {os.path.exists(output_directory)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
        print("MFA alignment completed successfully.")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.stderr}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")


