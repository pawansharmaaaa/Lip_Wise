import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

import helpers.file_check as fc
import gradio as gr


class FaceResolver:
    def __init__(self):
        self.video_path = None
        self.face_detector = FaceAnalysis(providers=['CUDAExecutionProvider'])
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))

        self.frames = []
        self.fps = 0
        self.width = 0
        self.height = 0
        self.frame_count = 0
        
        self.selected_embedding = 0
        self.faces = []

        self.presence_mask = []
        self.bboxes = []
        self.kps = []

    def process_video(self, video_path) -> None:
        self.video_path = video_path
        video_capture = cv2.VideoCapture(self.video_path)
        
        self.fps = video_capture.get(cv2.CAP_PROP_FPS)
        self.width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frames.append(frame)
        video_capture.release()
        
        self.frames = frames
    
    def _get_faces(self, frame) -> list:
        faces = self.face_detector.get(frame)
        self.faces = faces
        face_images = []
        for face in faces:
            bbox = face.bbox.astype(int)
            face_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            face_images.append(face_image[:, :, ::-1])
        return face_images
    
    def update_frame(self, frame_index) -> np.ndarray:
        if frame_index < len(self.frames):
            frame = self.frames[frame_index]
            faces = self._get_faces(frame)
            return frame[:, :, ::-1], faces
        return None

    
    def save_embedding(self, index):
        self.selected_embedding = self.faces[index].normed_embedding

    def _compare_embeddings(self, emb1, emb2, strength = 1.2):
        return np.linalg.norm(emb1 - emb2) < strength

    def get_preview(self, similarity_strength):
        p_bar = gr.Progress()
        bboxes = []
        presence_mask = []
        keypoints = []

        video = cv2.VideoCapture(self.video_path)

        # Metadata
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Writer Object for output video
        preview_path = os.path.join(fc.MEDIA_DIR, 'preview.mp4')
        writer = cv2.VideoWriter(preview_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Loop through the video
        p_bar.__call__((0, self.frame_count), desc=f"Processing Video", unit="Frames")
        iteration = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            faces = self.face_detector.get(frame)
            if faces:
                present = False
                for face in faces:
                    bbox = face.bbox.astype(int)
                    landmark = face.kps.astype(int)
                    embedding = face.normed_embedding
                    if self._compare_embeddings(embedding, self.selected_embedding, similarity_strength):
                        bboxes.append(bbox)
                        keypoints.append(landmark)
                        present = True

                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        for i in range(5):
                            cv2.circle(frame, (landmark[i][0], landmark[i][1]), 1, (0, 0, 255), 2)
                presence_mask.append(present)
            else:
                presence_mask.append(False)
            
            p_bar.__call__((iteration+1, self.frame_count), unit="Frames")
            iteration += 1
            writer.write(frame)
        
        self.presence_mask = np.asarray(presence_mask, dtype=bool)
        self.bboxes = np.asarray(bboxes)
        self.kps = np.asarray(keypoints)
        
        video.release()
        writer.release()
        
        return preview_path

    def save_state(self) -> None:
        gr.Info("Saving Data")
        np.save(os.path.join(fc.NPY_FILES_DIR, 'presence_mask.npy'), self.presence_mask)
        np.save(os.path.join(fc.NPY_FILES_DIR, 'bboxes.npy'), self.bboxes)
        np.save(os.path.join(fc.NPY_FILES_DIR, 'keypoints.npy'), self.kps)

