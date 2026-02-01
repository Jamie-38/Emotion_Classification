# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:45:46 2024

@author: Jayyy
"""
import imageio_ffmpeg as ffmpeg
import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
import subprocess


class AudioController:
    """
    A class to handle audio extraction, conversion, and mel spectrogram generation.
    """
    def __init__(self, audio_path=None, output_file=None):
        """
        Initialize an AudioController instance.
        
        Parameters:
        audio_path (str): The path to the audio file.
        output_file (str): The path to save the converted WAV file.
        """
        if audio_path:
            self.ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
            self.extracted_audio = self.extract_audio(audio_path)
            self.converted_audio = self.to_wav(self.extracted_audio, output_file)
            
            self.sr = 44100
            self.n_mels = 128
            self.hop_length = 512
            
            self.mel = self.melspectrogram(self.extracted_audio, self.sr, self.n_mels, self.hop_length)
            self.show_melspectrogram(self.mel, self.sr, self.hop_length)            
            
        else:
            print("unable to handle audio")
    
    def extract_audio(self, filename):
        """
        Extract audio from the given file using FFmpeg.
        
        Parameters:
        filename (str): The path to the file to extract audio from.
        
        Returns:
        np.ndarray: The extracted audio data.
        """
        command = [
            self.ffmpeg_exe,
            '-i', filename,
            '-f', 'f32le',
            '-acodec', 'pcm_f32le',
            '-ac', '1',
            '-ar', '44100',
            '-'
        ]
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"extract_audio - ffmpeg command failed with error: {err.decode('utf-8')}")
        
        return np.frombuffer(out, np.float32)
    
    def to_wav(self, input_data, output_file):
        """
         Convert audio data to WAV format using FFmpeg.
        
         Parameters:
         input_data (np.ndarray): The input audio data.
         output_file (str): The path to save the WAV file.
         """
        command = [
                self.ffmpeg_exe,
                '-f', 'f32le',  # input 32bit float little endian
                '-ar', '44100',  # input sample rate 44100 Hz
                '-ac', '1',  # input 1 channel (mono)
                '-i', '-',  # input file via pipe
                '-acodec', 'pcm_s32le',  # output 32bit PCM
                '-y',  # overwrite output file if it already exists
                output_file
            ]
            
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
        try:
            chunk_size = 4096  # Define a chunk size for writing
            for start in range(0, len(input_data), chunk_size):
                end = start + chunk_size
                process.stdin.write(input_data[start:end].tobytes())
            
            process.stdin.close()
            
            out, err = process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"to_wav - ffmpeg command failed with error: {err.decode('utf-8')}")
            
            print(f"Audio converted to WAV successfully and saved to {output_file}")
        
        except Exception as e:
            raise RuntimeError(f"to_wav - an error occurred: {str(e)}")
        
        finally:
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
            process.terminate()
        

    
    def melspectrogram(self, audio, sr, n_mels, hop_length):
        """
        Generate a mel spectrogram from the given audio data.
        
        Parameters:
        audio (np.ndarray): The input audio data.
        sr (int): The sample rate.
        n_mels (int): The number of mel bands.
        hop_length (int): The hop length.
        
        Returns:
        np.ndarray: The generated mel spectrogram.
        """
        mel = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length))
        return mel
    
    def show_melspectrogram(self, mel, sr, hop_length):
        """
        Display the mel spectrogram.
        
        Parameters:
        mel (np.ndarray): The mel spectrogram data.
        sr (int): The sample rate.
        hop_length (int): The hop length.
        """
        plt.figure(figsize=(14, 4))
        librosa.display.specshow(mel, sr=sr,hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.title('Log mel spectrogram')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.show()        

    def retrive_mel_segment(self, video_frame_timestamp_ms, video_frame_duration_ms):
        """
        Retrieve the mel spectrogram segment for a given video frame.
        
        Parameters:
        video_frame_timestamp_ms (int): The timestamp of the video frame in milliseconds.
        video_frame_duration_ms (int): The duration of the video frame in milliseconds.
        
        Returns:
        np.ndarray: The mel spectrogram segment for the video frame.
        """        
        segment_start_sec = video_frame_timestamp_ms / 1000.0
        segment_end_sec = (video_frame_timestamp_ms + video_frame_duration_ms) / 1000.0       

        mel_frame_start_index = int((segment_start_sec * self.sr) / self.hop_length)
        mel_frame_end_index = int((segment_end_sec * self.sr) / self.hop_length)
        
        mel_segment = self.mel[:, mel_frame_start_index:mel_frame_end_index]      
        
        return  mel_segment
    

