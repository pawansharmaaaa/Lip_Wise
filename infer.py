# This file is a part of https://github.com/pawansharmaaaa/Lip_Wise/ repository.

import os
import cv2
import torch
import subprocess
import platform

import numpy as np
import gradio as gr

from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Custom Modules
from helpers import audio
from helpers import file_check
from helpers import preprocess_mp as pmp
from helpers import model_loaders
from helpers import batch_processors

# Global Variables
TEMP_DIRECTORY = file_check.TEMP_DIR
MEDIA_DIRECTORY = file_check.MEDIA_DIR
NPY_FILES_DIRECTORY = file_check.NPY_FILES_DIR
OUTPUT_DIRECTORY = file_check.OUTPUT_DIR

#################################################### IMAGE INFERENCE ####################################################
def infer_image(frame_path, audio_path, pad, align_3d = False, 
                face_restorer = 'CodeFormer', 
                fps=30, 
                mel_step_size=16, 
                weight = 1.0, 
                upscale_bg = False, 
                bgupscaler='RealESRGAN_x4plus',
                gan=False):
    
    # Perform checks to ensure that all required files are present
    file_check.perform_check(bg_model_name=bgupscaler, restorer=face_restorer, use_gan_version=gan)

    # Get input type
    input_type, img_ext = file_check.get_file_type(frame_path)
    if input_type != "image":
        raise Exception("Input file is not an image. Try again with an image file.")
    
    # Get audio type
    audio_type, aud_ext = file_check.get_file_type(audio_path)
    if audio_type != "audio":
        raise Exception("Input file is not an audio.")
    if aud_ext != "wav":
        gr.Info("Audio file is not a wav file. Converting to wav...")
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

    # Create media_preprocess object and helper object
    processor = pmp.ModelProcessor(padding=pad)

    # Read image
    if align_3d:
      frame = cv2.imread(frame_path)
      frame = processor.align_3d(frame)
    else:
      frame = cv2.imread(frame_path)
    height, width, _ = frame.shape

    file_name = os.path.basename(frame_path).split('.')[0] + '_output.mp4'

    # Get face landmarks
    print("Getting face landmarks...")
    processor.detect_for_image(frame.copy())

    # Create face helper object from landmarks
    helper = pmp.FaceHelpers(image_mode=True)

    # Create progress bar
    p_bar = gr.Progress()

    # extract face from image
    print("Extracting face from image...")
    extracted_face, mask, inv_mask, center, bbox = helper.extract_face(original_img=frame)

    # Warp, Crop and Align face
    print("Warping, cropping and aligning face...")
    cropped_face, aligned_bbox, rotation_matrix = helper.align_crop_face(extracted_face=extracted_face)
    cropped_face_height, cropped_face_width, _ = cropped_face.shape

    total = pmp.Total_stat()
    # Generate data for inference
    print("Generating data for inference...")
    gen = helper.gen_data_image_mode(cropped_face, mel_chunks, total)

    print(f"Total mels:  {total.mels}")

    # Create model loader object
    ml = model_loaders.ModelLoader(face_restorer, weight)

    # Load wav2lip model
    w2l_model = ml.load_wav2lip_model(gan=gan)

    # Initialize video writer
    out = cv2.VideoWriter(os.path.join(MEDIA_DIRECTORY, 'temp.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("Processing.....")
    batch_no = 1
    # Feed to model:
    for (img_batch, mel_batch) in gen:
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            dubbed_faces = w2l_model(mel_batch, img_batch)
        
        dubbed_faces = dubbed_faces.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        if face_restorer == 'CodeFormer':
            with ThreadPoolExecutor() as executor:
                restored_faces = list(executor.map(ml.restore_wCodeFormer, dubbed_faces))
        elif face_restorer == 'GFPGAN':
            with ThreadPoolExecutor() as executor:
                restored_faces = list(executor.map(ml.restore_wGFPGAN, dubbed_faces))
        elif face_restorer == 'RestoreFormer':
            with ThreadPoolExecutor() as executor:
                restored_faces = list(executor.map(ml.restore_wRF, dubbed_faces))

        for face in restored_faces:
            processed_face = cv2.resize(face, (cropped_face_width, cropped_face_height), interpolation=cv2.INTER_LANCZOS4)
            processed_ready = helper.paste_back_black_bg(processed_face, aligned_bbox, frame, ml)
            ready_to_paste = helper.unwarp_align(processed_ready, rotation_matrix)
            final = helper.paste_back(ready_to_paste, frame, mask, inv_mask, center)

            if upscale_bg:
                final, _ = ml.restore_background(final, bgupscaler, tile=400, outscale=1.0, half=False)

            out.write(final)

        p_bar.__call__((batch_no, total.mels+1))
        batch_no += 1
                
    out.release()
    del gen
    command = f"ffmpeg -y -i {audio_path} -i {os.path.join(MEDIA_DIRECTORY, 'temp.mp4')} -strict -2 -q:v 1 {os.path.join(OUTPUT_DIRECTORY, file_name)}"
    subprocess.call(command, shell=platform.system() != 'Windows')

    p_bar.__call__((batch_no, total.mels+1))

    gr.Info(f"Done! Check {file_name} in output directory.")

    return os.path.join(OUTPUT_DIRECTORY, file_name)

################################################## VIDEO INFERENCE ##################################################

def infer_video(video_path, audio_path, pad, 
                face_restorer='CodeFormer',
                mel_step_size=16, 
                weight = 1.0, 
                upscale_bg = False, 
                bgupscaler='RealESRGAN_x2plus',
                gan=False):
    # Perform checks to ensure that all required files are present
    file_check.perform_check(model_name=bgupscaler, restorer=face_restorer, use_gan_version=gan)

    # Get input type
    input_type, vid_ext = file_check.get_file_type(video_path)
    if input_type != "video":
        raise Exception("Input file is not a video. Try again with an video file.")
    
    # Get audio type
    audio_type, aud_ext = file_check.get_file_type(audio_path)
    if audio_type != "audio":
        raise Exception("Input file is not an audio.")
    if aud_ext != "wav":
        gr.Info("Audio file is not a wav file. Converting to wav...")
        # Convert audio to wav
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, os.path.join(MEDIA_DIRECTORY, 'aud_input.wav'))
        subprocess.call(command, shell=True)
        audio_path = os.path.join(MEDIA_DIRECTORY, 'aud_input.wav')

    # Create media_preprocess object and helper object
    processor = pmp.ModelProcessor(padding=pad)

    # Get face landmarks
    print("Getting face landmarks...")
    processor.detect_for_video(video_path)
    
    # Check for cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} for inference.')

    # Generate audio spectrogram
    print("Generating audio spectrogram...")
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)

    # Load video
    video = cv2.VideoCapture(video_path)
    
    file_name = os.path.basename(video_path).split('.')[0] + '_output.mp4'

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create mel chunks array
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

    if len(mel_chunks) > frame_count:
        gr.Info("Audio is longer than video. Truncating audio...")
        mel_chunks = mel_chunks[:frame_count]
    elif len(mel_chunks) < frame_count:
        gr.Info("Video is longer than audio. Truncating video...")
        frame_count = len(mel_chunks)

    # Creating Boolean mask
    total_frames = np.arange(0, frame_count) 
    no_face_index = np.load(os.path.join(NPY_FILES_DIRECTORY, 'no_face_index.npy'))

    mask = np.isin(total_frames, no_face_index, invert=True).astype(bool)
    mask_batch = np.array_split(mask, len(mask)//16, axis=0)

    print(f"Length of mel chunks: {len(mel_chunks)}")

    # Split mel chunks into batches
    mel_chunks = np.array(mel_chunks)
    mel_chunks_batch = np.array_split(mel_chunks, len(mel_chunks)//16, axis=0)

    # Create an array of frame numbers and split it into batches
    frame_numbers = np.arange(0, frame_count)
    frame_nos_batch = np.array_split(frame_numbers, frame_count//16, axis=0)

    # Create batch helper object
    bp = batch_processors.BatchProcessors()

    # Create VideoWriter object
    writer = cv2.VideoWriter(os.path.join(MEDIA_DIRECTORY, 'temp.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Create model loader object
    ml = model_loaders.ModelLoader(face_restorer, weight)

    # Load wav2lip model
    w2l_model = ml.load_wav2lip_model(gan=gan)

    # Start image processing
    images = []
    batch_no = 0
    est_total_batches = len(mel_chunks_batch) + 1

    p_bar = gr.Progress()

    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame_no = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        images.append(frame)

        if len(images) == len(mask_batch[batch_no]):
            
            frames = np.array(images)
            frame_nos_to_input = frame_nos_batch[batch_no][mask_batch[batch_no]]
            mels_to_input = mel_chunks_batch[batch_no][mask_batch[batch_no]]
            frames_to_input = frames[mask_batch[batch_no]]
            
            if len(frames_to_input) != 0 and len(mels_to_input) != 0:
                extracted_faces, face_masks, inv_masks, centers, bboxes = bp.extract_face_batch(frames_to_input, frame_nos_to_input)
                cropped_faces, aligned_bboxes, rotation_matrices = bp.align_crop_batch(extracted_faces, frame_nos_to_input)
                frame_batch, mel_batch = bp.gen_data_video_mode(cropped_faces, mels_to_input)

                # Feed to wav2lip model:
                frame_batch = torch.FloatTensor(np.transpose(frame_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

                with torch.no_grad():
                    dubbed_faces = w2l_model(mel_batch, frame_batch)
                
                dubbed_faces = dubbed_faces.cpu().numpy().transpose(0, 2, 3, 1) * 255.

                if face_restorer == 'CodeFormer':
                    with ThreadPoolExecutor() as executor:
                        restored_faces = list(executor.map(ml.restore_wCodeFormer, dubbed_faces))
                elif face_restorer == 'GFPGAN':
                    with ThreadPoolExecutor() as executor:
                        restored_faces = list(executor.map(ml.restore_wGFPGAN, dubbed_faces))
                elif face_restorer == 'RestoreFormer':
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        restored_faces = list(executor.map(ml.restore_wRF, dubbed_faces))
                
                # Post processing
                resized_restored_faces = bp.face_resize_batch(restored_faces, cropped_faces)
                pasted_ready_faces = bp.paste_back_black_bg_batch(resized_restored_faces, aligned_bboxes, frames_to_input, ml)
                ready_to_paste = bp.unwarp_align_batch(pasted_ready_faces, rotation_matrices)
                restored_images = bp.paste_back_batch(ready_to_paste, frames_to_input, face_masks, inv_masks, centers)

                frames[mask_batch[batch_no]] = restored_images

            print(f"Writing batch no: {batch_no} out of total {est_total_batches} batches.")
            for frame in frames:
                if upscale_bg:
                    frame, _ = ml.restore_background(frame, bgupscaler, tile=400, outscale=1.0, half=False)
                writer.write(frame)
            batch_no += 1
            p_bar.__call__((batch_no, est_total_batches))

            images = []

            if batch_no == len(mel_chunks_batch):
                gr.Info("Reached end of Audio, Video has been dubbed.")
                break

    video.release()
    writer.release()

    command = f"ffmpeg -y -i {audio_path} -i {os.path.join(MEDIA_DIRECTORY, 'temp.mp4')} -strict -2 -q:v 1 -shortest {os.path.join(OUTPUT_DIRECTORY, file_name)}"
    subprocess.call(command, shell=platform.system() != 'Windows')
    p_bar.__call__((est_total_batches, est_total_batches))

    return os.path.join(OUTPUT_DIRECTORY, file_name)