# This file is a part of https://github.com/pawansharmaaaa/Lip_Wise/ repository.

import cv2
import math
import os
import asyncio

import mediapipe as mp
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import preprocess_mp
import file_check

class BatchProcessors:
    def __init__(self):

        self.npy_directory = file_check.NPY_FILES_DIR
        self.weights_directory = file_check.WEIGHTS_DIR
        self.video_landmarks_path = os.path.join(self.npy_directory,'video_landmarks.npy')

        try:
            self.landmarks_all = np.load(self.video_landmarks_path)
        except FileNotFoundError as e:
            print("Video landmarks were not saved. Please report this issue.")
            exit(1)

        self.helper = preprocess_mp.FaceHelpers()


    def extract_face_batch(self, frame_batch, frame_numbers):
        with ThreadPoolExecutor() as executor:
            extracted_faces, original_masks = zip(*list(executor.map(self.helper.extract_face, frame_batch, frame_numbers)))
        return extracted_faces, original_masks
    
    def alignment_procedure_batch(self, extracted_faces, frame_numbers):
        with ThreadPoolExecutor() as executor:
            aligned_faces, rotation_matrices = zip(*list(executor.map(self.helper.alignment_procedure, extracted_faces, frame_numbers)))
        return aligned_faces, rotation_matrices
    
    def crop_extracted_face_batch(self, aligned_faces, rotation_matrices, frame_numbers):
        with ThreadPoolExecutor() as executor:
            cropped_faces, bboxes = zip(*list(executor.map(self.helper.crop_extracted_face, aligned_faces, rotation_matrices, frame_numbers)))
        return cropped_faces, bboxes
    
    def gen_data_video_mode(self, cropped_faces_batch, mel_batch):
        """
        Generates data for inference in video mode.
        Batches the data to be fed into the model.
        Batch of image includes several images of shape (96, 96, 6) stacked together.
        These images contain the half face and the full face.

        Args:
            cropped_faces: a batch of size batch_size of The cropped faces obtained from the crop_extracted_face function.
            mel_batch: a batch of size batch_size consisting of The mel chunks obtained from the audio.
        
        Returns:
            A batch of images of shape (96, 96, 6) and mel chunks.
        """
        resized_cropped_faces_batch = []
        # Resize face for wav2lip
        for cropped_face in cropped_faces_batch:
            cropped_face = cv2.resize(cropped_face, (96, 96))
            resized_cropped_faces_batch.append(cropped_face)

        frame_batch = np.asarray(resized_cropped_faces_batch)

        img_masked = frame_batch.copy()
        img_masked[:, 96//2:] = 0

        frame_batch = np.concatenate((img_masked, frame_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        return frame_batch, mel_batch

    def face_resize_batch(self, restored_faces, cropped_faces_batch):
        size_batch = []
        for cropped_face in cropped_faces_batch:
            height, width = cropped_face.shape[:2]
            size_batch.append((width, height))

        resizer_partial = partial(cv2.resize, interpolation=cv2.INTER_LANCZOS4)
        with ThreadPoolExecutor() as executor:
            resized_restored_faces = list(executor.map(resizer_partial, restored_faces, size_batch))
        return resized_restored_faces

    def paste_back_black_bg_batch(self, processed_face_batch, bboxes_batch, frame_batch):
        with ThreadPoolExecutor() as executor:
            pasted_ready_faces = list(executor.map(self.helper.paste_back_black_bg, processed_face_batch, bboxes_batch, frame_batch))
        return pasted_ready_faces
    
    def unwarp_align_batch(self, pasted_ready_faces, rotation_matrices):
        with ThreadPoolExecutor() as executor:
            ready_to_paste = list(executor.map(self.helper.unwarp_align, pasted_ready_faces, rotation_matrices))
        return ready_to_paste
    
    def paste_back_batch(self, ready_to_paste, frame_batch, original_masks):
        with ThreadPoolExecutor() as executor:
            pasted_faces = list(executor.map(self.helper.paste_back, ready_to_paste, frame_batch, original_masks))
        return pasted_faces