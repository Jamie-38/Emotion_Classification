# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:22:09 2024

@author: Jayyy
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np

class FaceLandMarkGenerator:
    """
    A class to generate and draw facial landmarks using Mediapipe.
    """
    def __init__(self, model_path):
        """
        Initialize a FaceLandMarkGenerator instance.
        
        Parameters:
        model_path (str): The path to the facial landmark model.
        """
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            running_mode=self.VisionRunningMode.VIDEO
        )
        
        self.landmarker = self.FaceLandmarker.create_from_options(self.options)       

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
    
    def draw_landmarks(self, frame, face_landmarks_list):
        """
        Draw facial landmarks on the given frame.
        
        Parameters:
        frame (np.ndarray): The video frame to draw landmarks on.
        face_landmarks_list (list): List of facial landmarks to draw.
        """
        for face_landmarks in face_landmarks_list:
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])
            
            # Draw tesselation, contours, and irises on the frame
            solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
            solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list= face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
            solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
            )
    
    def find_landmarks(self, frame, frame_timestamp_ms):
        """
        Find facial landmarks in the given frame.
        
        Parameters:
        frame (np.ndarray): The video frame to analyze.
        frame_timestamp_ms (int): The timestamp of the frame in milliseconds.
        
        Returns:
        list: List of detected facial landmarks.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_landmarker_result = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        face_landmarks_list = face_landmarker_result.face_landmarks
        return face_landmarks_list

        