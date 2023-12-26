import os
import cv2
import torch
import subprocess
import platform

import numpy as np
import gradio as gr

from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize

# Custom Modules
import audio
import file_check
import preprocess_mp as pmp
import model_loaders as ml

# Global Variables
TEMP_DIRECTORY = file_check.TEMP_DIR
MEDIA_DIRECTORY = file_check.MEDIA_DIR
NPY_FILES_DIRECTORY = file_check.NPY_FILES_DIR
OUTPUT_DIRECTORY = file_check.OUTPUT_DIR

# Image Inference
def infer_image(frame_path, audio_path, fps=30, mel_step_size=16, gfpgan_weight = 1.0):
    
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
        processor.preprocess_image(frame.copy())

        # Create face helper object from landmarks
        helper = pmp.FaceHelpers(image_mode=True)

        # extract face from image
        print("Extracting face from image...")
        extracted_face, original_mask = helper.extract_face(frame.copy())

        # warp and align face
        print("Warping and aligning face...")
        aligned_face, rotation_matrix = helper.warp_align_extracted_face(extracted_face)

        # Crop face
        print("Cropping face...")
        cropped_face, bbox = helper.crop_extracted_face(aligned_face, rotation_matrix)

        # Store cropped face's height and width
        cropped_face_height, cropped_face_width, _ = cropped_face.shape

        # Generate data for inference
        gen = helper.gen_data_image_mode(cropped_face, mel_chunks)

        # Load wav2lip model
        w2l_model = ml.load_wav2lip_gan_model()

        # Load GFPGAN model
        gfpgan = ml.load_gfpgan_model()
        gfpgan = gfpgan.to(device)

        # Initialize video writer
        out = cv2.VideoWriter(os.path.join(MEDIA_DIRECTORY, 'temp.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Feed to model:
        for i, (img_batch, mel_batch) in enumerate(gr.Progress(track_tqdm=True).tqdm(gen)):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                dubbed_faces = w2l_model(mel_batch, img_batch)
            
            dubbed_faces = dubbed_faces.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for d in dubbed_faces:
                d = cv2.resize(d.astype(np.uint8) / 255., (512, 512), interpolation=cv2.INTER_CUBIC)
                dubbed_face_t = img2tensor(d, bgr2rgb=True, float32=True)
                normalize(dubbed_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                dubbed_face_t = dubbed_face_t.unsqueeze(0).to(device)
                
                try:
                    output = gfpgan(dubbed_face_t, return_rgb=False, weight=gfpgan_weight)[0]
                    restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
                except RuntimeError as error:
                    print(f'\tFailed inference for GFPGAN: {error}.')
                    restored_face = d
                
                restored_face = restored_face.astype(np.uint8)

                # Warp face back to original pose
                
                processed_face = cv2.resize(restored_face, (cropped_face_width, cropped_face_height), interpolation=cv2.INTER_LANCZOS4)
                processed_ready = helper.paste_back_black_bg(processed_face, bbox, extracted_face)
                ready_to_paste = helper.unwarp_align(processed_ready, rotation_matrix)
                restored_image = helper.paste_back(ready_to_paste, frame, original_mask)

                out.write(restored_image)
            
        out.release()

        command = f"ffmpeg -y -i {audio_path} -i {os.path.join(MEDIA_DIRECTORY, 'temp.mp4')} -strict -2 -q:v 1 {os.path.join(OUTPUT_DIRECTORY, 'output.mp4')}"
        subprocess.call(command, shell=platform.system() != 'Windows')

        return os.path.join(OUTPUT_DIRECTORY, 'output.mp4')

if __name__ == "__main__":
    # Create interface
    inputs = [
        gr.Image(type="filepath", label="Image"),
        gr.Audio(type="filepath", label="Audio"),
        gr.Slider(minimum=1, maximum=60, step=1, value=30, label="FPS"),
        gr.Slider(minimum=0, maximum=160, step=16, value=16, label="Mel Step Size"),
        gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="GFPGAN Weight")
    ]
    outputs = gr.Video(sources='upload', label="Output")
    title = "Lip Wise"

    gr.Interface(fn=infer_image, inputs=inputs, outputs=outputs, title=title).launch()
    
