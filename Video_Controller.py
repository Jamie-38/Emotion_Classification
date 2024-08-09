# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:53:13 2024

@author: Jayyy
"""
import cv2
import numpy as np


class VideoController:
    """
    A class to handle video processing tasks such as reading frames
    and retrieving frame timestamps.
    """
    def __init__(self, video_path):
        """
        Initialize a VideoController instance.
        
        Parameters:
        video_path (str): The path to the video file.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_duration_ms = 1000 / self.fps # Duration of each frame in milliseconds
        self.frame_index = 0
    
    def process_video(self):
        """
        Generator function to read and yield frames from the video along with timestamps.
        
        Yields:
        tuple: A tuple containing the frame, frame timestamp in milliseconds, and frame index.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_timestamp_ms = int(self.frame_index * self.frame_duration_ms)   
            self.frame_index += 1
            yield frame, frame_timestamp_ms, self.frame_index
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def show_frame(self, frame):
        """
        Display a video frame.
        
        Parameters:
        frame (np.ndarray): The video frame to display.
        """
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()

