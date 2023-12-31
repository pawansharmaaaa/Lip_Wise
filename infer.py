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
import audio
import file_check
import preprocess_mp as pmp
import model_loaders
import batch_processors

# Global Variables
TEMP_DIRECTORY = file_check.TEMP_DIR
MEDIA_DIRECTORY = file_check.MEDIA_DIR
NPY_FILES_DIRECTORY = file_check.NPY_FILES_DIR
OUTPUT_DIRECTORY = file_check.OUTPUT_DIR

#################################################### IMAGE INFERENCE ####################################################
def infer_image(frame_path, audio_path, face_restorer = 'CodeFormer', fps=30, mel_step_size=16, weight = 1.0):
    
    # Perform checks to ensure that all required files are present
    file_check.perform_check()

    # Get input type
    input_type, img_ext = file_check.get_file_type(frame_path)
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

    # Create media_preprocess object and helper object
    processor = pmp.model_processor()

    # Read image
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape

    # Get face landmarks
    print("Getting face landmarks...")
    processor.detect_for_image(frame.copy())

    # Create face helper object from landmarks
    helper = pmp.FaceHelpers(image_mode=True)

    # extract face from image
    print("Extracting face from image...")
    extracted_face, original_mask = helper.extract_face(frame.copy())

    # warp and align face
    print("Warping and aligning face...")
    aligned_face, rotation_matrix = helper.alignment_procedure(frame.copy())

    # Crop face
    print("Cropping face...")
    cropped_face, bbox = helper.crop_extracted_face(aligned_face, rotation_matrix)

    # Store cropped face's height and width
    cropped_face_height, cropped_face_width, _ = cropped_face.shape

    # Generate data for inference
    print("Generating data for inference...")
    gen = helper.gen_data_image_mode(cropped_face, mel_chunks)

    # Create model loader object
    ml = model_loaders.model_loaders(face_restorer, weight)

    # Load wav2lip model
    w2l_model = ml.load_wav2lip_model()

    # Initialize video writer
    out = cv2.VideoWriter(os.path.join(MEDIA_DIRECTORY, 'temp.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Feed to model:
    for (img_batch, mel_batch) in gr.Progress(track_tqdm=True).tqdm(gen, total=len(mel_chunks)):
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
        
        for face in restored_faces:
            processed_face = cv2.resize(face, (cropped_face_width, cropped_face_height), interpolation=cv2.INTER_LANCZOS4)
            processed_ready = helper.paste_back_black_bg(processed_face, bbox, frame)
            ready_to_paste = helper.unwarp_align(processed_ready, rotation_matrix)
            final = helper.paste_back(ready_to_paste, frame, original_mask)
            
            # Write each processed face to `out`
            out.write(final)
        
    out.release()

    command = f"ffmpeg -y -i {audio_path} -i {os.path.join(MEDIA_DIRECTORY, 'temp.mp4')} -strict -2 -q:v 1 {os.path.join(OUTPUT_DIRECTORY, 'output.mp4')}"
    subprocess.call(command, shell=platform.system() != 'Windows')

    return os.path.join(OUTPUT_DIRECTORY, 'output.mp4')

################################################## VIDEO INFERENCE ##################################################

def infer_video(video_path, audio_path, face_restorer='CodeFormer',mel_step_size=16, weight = 1.0):
    # Perform checks to ensure that all required files are present
    file_check.perform_check()

    # Get input type
    input_type, vid_ext = file_check.get_file_type(video_path)
    if input_type != "video":
        raise Exception("Input file is not a video. Try again with an video file.")
    
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

    # Create media_preprocess object and helper object
    processor = pmp.model_processor()

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
        print("Audio is longer than video. Truncating audio...")
        mel_chunks = mel_chunks[:frame_count]
    elif len(mel_chunks) < frame_count:
        print("Video is longer than audio. Truncating video...")
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
    ml = model_loaders.model_loaders(face_restorer, weight)

    # Load wav2lip model
    w2l_model = ml.load_wav2lip_model()

    # Start image processing
    images = []
    batch_no = 0
    est_total_batches = len(mel_chunks_batch)

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
                extracted_faces, original_masks = bp.extract_face_batch(frames_to_input, frame_nos_to_input)
                aligned_faces, rotation_matrices = bp.alignment_procedure_batch(frames_to_input, frame_nos_to_input)
                cropped_faces, bboxes = bp.crop_extracted_face_batch(aligned_faces, rotation_matrices, frame_nos_to_input)
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
                
                # Post processing
                resized_restored_faces = bp.face_resize_batch(restored_faces, cropped_faces)
                pasted_ready_faces = bp.paste_back_black_bg_batch(resized_restored_faces, bboxes, frames_to_input)
                ready_to_paste = bp.unwarp_align_batch(pasted_ready_faces, rotation_matrices)
                restored_images = bp.paste_back_batch(ready_to_paste, frames_to_input, original_masks)

                frames[mask_batch[batch_no]] = restored_images

            print(f"Writing batch no: {batch_no} out of total {est_total_batches} batches.")
            for frame in frames:
                writer.write(frame)
            batch_no += 1
            p_bar.__call__((batch_no, est_total_batches))

            images = []

            if batch_no == len(mel_chunks_batch):
                print("Reached end of Audio, Video has been dubbed.")
                break

    video.release()
    writer.release()

    command = f"ffmpeg -y -i {audio_path} -i {os.path.join(MEDIA_DIRECTORY, 'temp.mp4')} -strict -2 -q:v 1 {os.path.join(OUTPUT_DIRECTORY, 'output.mp4')}"
    subprocess.call(command, shell=platform.system() != 'Windows')

    return os.path.join(OUTPUT_DIRECTORY, 'output.mp4')