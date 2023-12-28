import cv2
import math
import os
import asyncio

import mediapipe as mp
import numpy as np

from concurrent.futures import ThreadPoolExecutor

import preprocess_mp
import file_check

class batch_processors:
    def __init__(self, big_frame_batch, start_idx, end_idx, batch_size):

        self.frame_numbers = np.arange(start_idx, end_idx)
        self.batch_size = batch_size

        self.big_frame_batch = big_frame_batch
        self.frame_batch = np.array_split(self.big_frame_batch, 
                                        len(self.big_frame_batch)//self.batch_size,
                                        axis=0)

        self.npy_directory = file_check.NPY_FILES_DIR
        self.weights_directory = file_check.WEIGHTS_DIR
        self.video_landmarks_path = os.path.join(self.npy_directory,'video_landmarks.npy')

        try:
            self.landmarks_all = np.load(self.video_landmarks_path)
        except FileNotFoundError as e:
            print("Video landmarks were not saved. Please report this issue.")
            exit(1)


    def extract_face_batch(self, frame_batch, frame_numbers):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            extracted_faces, original_masks = list(executor.map(preprocess_mp.extract_face, frame_batch, frame_numbers))
        return extracted_faces, original_masks
    
    def alignment_procedure_batch(self, extracted_faces, frame_numbers):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            aligned_faces, rotation_matrices = list(executor.map(preprocess_mp.alignment_procedure, extracted_faces, frame_numbers))
        return aligned_faces, rotation_matrices
    
    def crop_extracted_face_batch(self, aligned_faces, rotation_matrices, frame_numbers):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            cropped_faces, bboxes = list(executor.map(preprocess_mp.crop_extracted_face, aligned_faces, rotation_matrices, frame_numbers))
        return cropped_faces, bboxes
    
    def paste_back_black_bg_batch