# This file is a part of https://github.com/pawansharmaaaa/Lip_Wise/ repository.

import os
import cv2
import torch
import subprocess
import platform

import numpy as np
import gradio as gr

from concurrent.futures import ThreadPoolExecutor

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
@torch.no_grad()
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
    free_memory = torch.cuda.mem_get_info()[0]
    print(f"Initial Free Memory: {free_memory/1024**3:.2f} GB")

    # Limiting the number of threads to avoid vram issues
    limit = free_memory // 2e9

    # Do not use GPU if free memory is less than 2GB
    device = 'cuda' if torch.cuda.is_available() and limit!=0 else 'cpu'
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

    file_name_check = os.path.basename(frame_path).split('.')
    if len(file_name_check) > 2:
        return "Please remove unneccesary periods('.') from file name and try again."
    else:
        file_name = file_name_check[0] + '_output.mp4'

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

    # Create model loader object
    ml = model_loaders.ModelLoader(face_restorer, weight)

    # Load wav2lip model
    w2l_model = ml.load_wav2lip_model(gan=gan)

    # Initialize video writer
    out = cv2.VideoWriter(os.path.join(MEDIA_DIRECTORY, 'temp.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("Processing.....")
    p_bar.__call__(0, desc=f"Initializing Lip Sync...")
    batch_no = 1
    # Feed to model:
    for (img_batch, mel_batch) in gen:
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            dubbed_faces = w2l_model(mel_batch, img_batch)
        
        dubbed_faces = dubbed_faces.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        if face_restorer == 'CodeFormer':
            with ThreadPoolExecutor(max_workers=limit) as executor:
                restored_faces = list(executor.map(ml.restore_wCodeFormer, dubbed_faces))
        elif face_restorer == 'GFPGAN':
            with ThreadPoolExecutor(max_workers=limit) as executor:
                restored_faces = list(executor.map(ml.restore_wGFPGAN, dubbed_faces))
        elif face_restorer == 'RestoreFormer':
            with ThreadPoolExecutor(max_workers=limit) as executor:
                restored_faces = list(executor.map(ml.restore_wRF, dubbed_faces))
        elif face_restorer == "None":
            restored_faces = dubbed_faces
        else:
            raise Exception("Invalid face restorer model. Please check the model name and try again.")

        up_progress = gr.Progress()
        for idx, face in enumerate(restored_faces):
            processed_face = cv2.resize(face, (cropped_face_width, cropped_face_height), interpolation=cv2.INTER_LANCZOS4)
            processed_ready = helper.paste_back_black_bg(processed_face, aligned_bbox, frame, ml)
            ready_to_paste = helper.unwarp_align(processed_ready, rotation_matrix)
            final = helper.paste_back(ready_to_paste, frame, mask, inv_mask, center)

            if upscale_bg:
                up_progress.__call__((idx, len(restored_faces)), desc=f"Upscaling frame: {idx} out of {len(restored_faces)} in batch: {batch_no}/{total.mels}")
                final, _ = ml.restore_background(final, bgupscaler, tile=400, outscale=1.0, half=False)

            out.write(final)
        p_bar.__call__((batch_no, total.mels), desc=f"Processed batch: {batch_no} out of {total.mels}")
        batch_no += 1
                
    out.release()
    del gen
    
    p_bar2 = gr.Progress()
    p_bar2.__call__((25, 100), desc=f"Merging audio and video...")
    
    command = f"ffmpeg -y -i {audio_path} -i {os.path.join(MEDIA_DIRECTORY, 'temp.mp4')} -strict -2 -q:v 1 {os.path.join(OUTPUT_DIRECTORY, file_name)}"
    subprocess.call(command, shell=platform.system() != 'Windows')

    p_bar2.__call__((100, 100), desc=f"Done!")

    gr.Info(f"Done! Check {file_name} in output directory.")

    return os.path.join(OUTPUT_DIRECTORY, file_name)

################################################## VIDEO INFERENCE ##################################################

@torch.no_grad()
def infer_video(video_path, audio_path, pad, 
                face_restorer='CodeFormer',
                mel_step_size=16, 
                weight = 1.0, 
                upscale_bg = False, 
                bgupscaler='RealESRGAN_x2plus',
                gan=False,
                loop=False):
    # Perform checks to ensure that all required files are present
    file_check.perform_check(bg_model_name=bgupscaler, restorer=face_restorer, use_gan_version=gan)

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

    if loop:
        gr.Info("Looping video...")
        video_path = processor.loop_video(video_path=video_path, audio_path=audio_path)
    else:
        pass

    # Get face landmarks
    gr.Info("Getting face landmarks...")
    processor.detect_for_video(video_path)
    
    # Check for cuda
    free_memory = torch.cuda.mem_get_info()[0]
    gr.Info(f"Initial Free Memory: {free_memory/1024**3:.2f} GB")

    # Limiting the number of threads to avoid vram issues
    limit = free_memory // 2e9

    # Do not use GPU if free memory is less than 2GB
    device = 'cuda' if torch.cuda.is_available() and limit!=0 else 'cpu'
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
    # The mel_idx_multiplier aligns audio chunks with video frames for consistent processing and analysis.
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
    est_total_batches = len(mel_chunks_batch)

    p_bar = gr.Progress()
    p_bar.__call__((0, est_total_batches), desc=f"Initializing Lip Sync...")

    # Create a directory inside the data directory
    data_directory = "data"
    list_directory = os.path.join(data_directory, "lists")
    os.makedirs(list_directory, exist_ok=True)

    # Create directories with the name of each list
    lists = ["images_list", "mels_list", "extracted_faces_list", "face_masks_list", "inv_masks_list", "cropped_faces_list", "frame_batch_list", "mel_batch_list", "dubbed_faces_list", "restored_faces_list", "resized_restored_faces_list", "pasted_ready_faces_list", "ready_to_paste_list", "restored_images_list", "upscaled_bg_list"]

    for list_name in lists:
        list_path = os.path.join(list_directory, list_name)
        os.makedirs(list_path, exist_ok=True)

    mels_list = []
    mel_batch_list = []
    frame_batch_list = []

    while batch_no<24:
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
                cv2.imwrite(os.path.join(list_directory, lists[0], f'image{batch_no}.png'),frames_to_input[0])
                mels_list.append(mels_to_input[0])
                
                extracted_faces, face_masks, inv_masks, centers, bboxes = bp.extract_face_batch(frames_to_input, frame_nos_to_input)

                # Save Data
                cv2.imwrite(os.path.join(list_directory, lists[2], f'image{batch_no}.png'), extracted_faces[0])
                cv2.imwrite(os.path.join(list_directory, lists[3], f'image{batch_no}.png'), face_masks[0])
                cv2.imwrite(os.path.join(list_directory, lists[4], f'image{batch_no}.png'), inv_masks[0])


                cropped_faces, aligned_bboxes, rotation_matrices = bp.align_crop_batch(extracted_faces, frame_nos_to_input)
                
                # Save Data
                cv2.imwrite(os.path.join(list_directory, lists[5], f'image{batch_no}.png'), cropped_faces[0])


                frame_batch, mel_batch = bp.gen_data_video_mode(cropped_faces, mels_to_input)
                
                # Save Data
                frame_batch_list.append(frame_batch[0])
                mel_batch_list.append(mel_batch[0])


                # Feed to wav2lip model:
                frame_batch = torch.FloatTensor(np.transpose(frame_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

                with torch.no_grad():
                    dubbed_faces = w2l_model(mel_batch, frame_batch)
                
                dubbed_faces = dubbed_faces.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                # Save Data
                cv2.imwrite(os.path.join(list_directory, lists[8], f'image{batch_no}.png'), dubbed_faces[0])

                if face_restorer == 'CodeFormer':
                    with ThreadPoolExecutor(max_workers=limit) as executor:
                        restored_faces = list(executor.map(ml.restore_wCodeFormer, dubbed_faces))
                elif face_restorer == 'GFPGAN':
                    with ThreadPoolExecutor(max_workers=limit) as executor:
                        restored_faces = list(executor.map(ml.restore_wGFPGAN, dubbed_faces))
                elif face_restorer == 'RestoreFormer':
                    with ThreadPoolExecutor(max_workers=limit) as executor:
                        restored_faces = list(executor.map(ml.restore_wRF, dubbed_faces))
                elif face_restorer == "None":
                    restored_faces = dubbed_faces
                else:
                    raise Exception("Invalid face restorer model. Please check the model name and try again.")
                
                # Save Data
                cv2.imwrite(os.path.join(list_directory, lists[9], f'image{batch_no}.png'), restored_faces[0])
                
                # Post processing
                resized_restored_faces = bp.face_resize_batch(restored_faces, cropped_faces)
                cv2.imwrite(os.path.join(list_directory, lists[10], f'image{batch_no}.png'), resized_restored_faces[0])
                pasted_ready_faces = bp.paste_back_black_bg_batch(resized_restored_faces, aligned_bboxes, frames_to_input, ml)
                cv2.imwrite(os.path.join(list_directory, lists[11], f'image{batch_no}.png'), pasted_ready_faces[0])
                ready_to_paste = bp.unwarp_align_batch(pasted_ready_faces, rotation_matrices)
                cv2.imwrite(os.path.join(list_directory, lists[12], f'image{batch_no}.png'), ready_to_paste[0])
                restored_images = bp.paste_back_batch(ready_to_paste, frames_to_input, face_masks, inv_masks, centers)
                cv2.imwrite(os.path.join(list_directory, lists[13], f'image{batch_no}.png'), restored_images[0])

                frames[mask_batch[batch_no]] = restored_images

            up_progress = gr.Progress()
            for idx, frame in enumerate(frames):
                if upscale_bg and idx==0:
                    up_progress.__call__((idx, len(frames)), desc=f"Upscaling frame: {idx} out of {len(restored_faces)} in batch: {batch_no}/{est_total_batches}")
                    frame, _ = ml.restore_background(frame, bgupscaler, tile=400, outscale=1.0, half=False)
                    cv2.imwrite(os.path.join(list_directory, lists[14], f'image{batch_no}.png'), frame)
                writer.write(frame)
            
            p_bar.__call__((batch_no+1, est_total_batches), desc=f"Processed batch: {batch_no} out of {est_total_batches}")
            print(f"Writing batch no: {batch_no+1} out of total {est_total_batches} batches.")
            batch_no += 1

            images = []

            if batch_no == len(mel_chunks_batch):
                gr.Info("Reached end of Audio, Video has been dubbed.")
                break
    
    mels_list = np.asarray(mels_list)
    mel_batch_list = np.asarray(mel_batch_list)
    frame_batch_list = np.asarray(frame_batch_list)

    np.save(os.path.join(list_directory, lists[1], 'mels_list.npy'), mels_list)
    np.save(os.path.join(list_directory, lists[6], 'frame_batch_list.npy'), frame_batch_list)
    np.save(os.path.join(list_directory, lists[7], 'mels_batch_list.npy'), mel_batch_list)

    video.release()
    writer.release()

    p_bar2 = gr.Progress()
    p_bar2.__call__((25, 100), desc=f"Merging audio and video...")
    
    command = f"ffmpeg -y -i {audio_path} -i {os.path.join(MEDIA_DIRECTORY, 'temp.mp4')} -strict -2 -q:v 1 -shortest {os.path.join(OUTPUT_DIRECTORY, file_name)}"
    subprocess.call(command, shell=platform.system() != 'Windows')
    
    p_bar2.__call__((100,100), desc="Merging audio and video...")

    return os.path.join(OUTPUT_DIRECTORY, file_name)