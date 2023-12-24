import os
import cv2
import torch
import subprocess

import gradio as gr

# Custom Modules
import audio
import file_check
import preprocess_mp as pmp

# Global Variables
TEMP_DIRECTORY = os.path.join(file_check.CURRENT_FILE_DIRECTORY, 'temp')
MEDIA_DIRECTORY = os.path.join(TEMP_DIRECTORY, 'media')
NPY_FILES_DIRECTORY = os.path.join(TEMP_DIRECTORY, 'npy_files')

# Image Inference
def infer_image(image_path, audio_path, fps=30, max_batch_size=16, mel_step_size=16, resize_factor=1):
    
        # Perform checks to ensure that all required files are present
        file_check.perform_check()

        # Create media_preprocess object and helper object
        pre_process = pmp.media_preprocess()
        helper = pmp.FaceHelpers()
    
        # Get input type
        input_type, img_ext = file_check.get_file_type(image_path)
        if input_type != "image":
            raise Exception("Input file is not an image. Try again with an image file.")
        
        # Get audio type
        audio_type, aud_ext = file_check.get_file_type(audio_path)
        if audio_type != "audio":
            raise Exception("Input file is not an audio.")
        if aud_ext != "wav":
            print("Audio file is not a wav file. Converting to wav...")
            # Convert audio to wav
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, os.path.join(MEDIA_DIRECTORY, 'aud_input.wav'))
            subprocess.call(command, shell=True)
            audio_path = os.path.join(MEDIA_DIRECTORY, 'aud_input.wav')
        
        # Check for cuda
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device} for inference.')

        # Generate audio spectrogram
        print("Generating audio spectrogram...")
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)

        mel_chunks = []
        #The mel_idx_multiplier aligns audio chunks with video frames for consistent processing and analysis.
        mel_idx_multiplier = 80./fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        print(f"Length of mel chunks: {len(mel_chunks)}")

        # fill frames array, it should have the same length as mel_chunks
        frames = [cv2.imread(image_path)]*len(mel_chunks)

