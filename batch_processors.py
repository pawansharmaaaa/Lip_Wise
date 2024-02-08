# This file is a part of https://github.com/pawansharmaaaa/Lip_Wise/ repository.

import cv2
import os

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

        self.helper = preprocess_mp.FaceHelpers()


    def extract_face_batch(self, frame_batch, frame_numbers):
        with ThreadPoolExecutor() as executor:
            extracted_faces, masks, inv_masks, centers, bboxes = zip(*list(executor.map(self.helper.extract_face, frame_batch, frame_numbers)))
        return extracted_faces, masks, inv_masks, centers, bboxes
    
    def align_crop_batch(self, extracted_faces, frame_numbers):
        with ThreadPoolExecutor() as executor:
            cropped_faces, aligned_bboxes, rotation_matrices = zip(*list(executor.map(self.helper.align_crop_face, extracted_faces, frame_numbers)))
        return cropped_faces, aligned_bboxes, rotation_matrices
    
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
            cropped_face = cv2.resize(cropped_face, (96, 96), interpolation=cv2.INTER_AREA)
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

    def paste_back_black_bg_batch(self, processed_face_batch, aligned_bboxes_batch, frame_batch):
        with ThreadPoolExecutor() as executor:
            pasted_ready_faces = list(executor.map(self.helper.paste_back_black_bg, processed_face_batch, aligned_bboxes_batch, frame_batch))
        return pasted_ready_faces
    
    def unwarp_align_batch(self, pasted_ready_faces, rotation_matrices):
        with ThreadPoolExecutor() as executor:
            ready_to_paste = list(executor.map(self.helper.unwarp_align, pasted_ready_faces, rotation_matrices))
        return ready_to_paste
    
    def paste_back_batch(self, ready_to_paste, frame_batch, face_masks, inv_masks, centers):
        with ThreadPoolExecutor() as executor:
            pasted_faces = list(executor.map(self.helper.paste_back, ready_to_paste, frame_batch, face_masks, inv_masks, centers))
        return pasted_faces
    
    # Partial Functions:
    def part_face_resize_batch(self, restored_faces, cropped_face):
        
        height, width = cropped_face.shape[:2]
        resizer_partial = partial(cv2.resize, dsize=(width,height),interpolation=cv2.INTER_LANCZOS4)
        with ThreadPoolExecutor() as executor:
            resized_restored_faces = list(executor.map(resizer_partial, restored_faces))
        return resized_restored_faces

    def part_paste_back_black_bg_batch(self, processed_face_batch, aligned_bbox, frame):
        func = partial(self.helper.paste_back_black_bg, aligned_bbox=aligned_bbox, full_frame=frame)
        with ThreadPoolExecutor() as executor:
            pasted_ready_faces = list(executor.map(func, processed_face_batch))
        return pasted_ready_faces
    
    def part_unwarp_align_batch(self, pasted_ready_faces, rotation_matrix):
        func = partial(self.helper.unwarp_align, rotation_matrix=rotation_matrix)
        with ThreadPoolExecutor() as executor:
            ready_to_paste = list(executor.map(func, pasted_ready_faces))
        return ready_to_paste
    
    def part_paste_back_batch(self, ready_to_paste, frame, face_mask, inv_mask, center):
        func = partial(self.helper.paste_back, original_img=frame, face_mask=face_mask, inv_mask=inv_mask, center=center)
        with ThreadPoolExecutor() as executor:
            pasted_faces = list(executor.map(func, ready_to_paste))
        return pasted_faces