# This file is a part of https://github.com/pawansharmaaaa/Lip_Wise/ repository.

import os
import gdown

import basicsr.archs as archs

from basicsr.utils.download_util import load_file_from_url

LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
DETECTOR_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'
GFPGAN_MODEL_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
CODEFORMERS_MODEL_URL = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
RESTOREFORMER_MODEL_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
WAV2LIP_MODEL_URL = ['https://drive.google.com/uc?id=1paYmN1KAZ2oPQPV-XauCoRUorhkOt0s2','https://drive.google.com/uc?id=1dhunIPYumA7WnR7dDsd7jsMgzgpOlg0V']
WAV2LIP_GAN_MODEL_URL = ['https://drive.google.com/uc?id=1WpqCULKQQcaCNf827h1qgjMHZENYHk-_','https://drive.google.com/uc?id=16UHRZv-oTW629AiMkSot5MrgDb42RJTX']
REAL_ESRGAN_MODEL_URL = {
            'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'RealESRNet_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
            'RealESRGAN_x4plus_anime_6B': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
            'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            'realesr-animevideov3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
            'realesr-general-wdn-x4v3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'realesr-general-x4v3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        }
VQFR2_MODEL_URL = 'https://github.com/TencentARC/VQFR/releases/download/v2.0.0/VQFR_v2.pth'
VQFR1_MODEL_URL = 'https://github.com/TencentARC/VQFR/releases/download/v1.0.0/VQFR_v1-33a1fac5.pth'

CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS_FILE_PATH = os.path.join(CURRENT_FILE_DIRECTORY, 'styles', 'style-black.css')

WEIGHTS_DIR = os.path.join(CURRENT_FILE_DIRECTORY, 'weights')
MP_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'mp')
GFPGAN_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'gfpgan')
CODEFORMERS_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'codeformers')
RESTOREFORMER_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'restoreformer')
WAV2LIP_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'wav2lip')
REALESRGAN_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'realesrgan')
VQFR_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'vqfr')

TEMP_DIR = os.path.join(CURRENT_FILE_DIRECTORY, 'temp')
NPY_FILES_DIR = os.path.join(TEMP_DIR, 'npy_files')
MEDIA_DIR = os.path.join(TEMP_DIR, 'media')

OUTPUT_DIR = os.path.join(CURRENT_FILE_DIRECTORY, 'output')

MP_LANDMARKER_MODEL_PATH = os.path.join(MP_WEIGHTS_DIR, 'face_landmarker.task')
MP_DETECTOR_MODEL_PATH = os.path.join(MP_WEIGHTS_DIR, 'blaze_face_short_range.tflite')
GFPGAN_MODEL_PATH = os.path.join(GFPGAN_WEIGHTS_DIR, 'GFPGANv1.4.pth')
CODEFORMERS_MODEL_PATH = os.path.join(CODEFORMERS_WEIGHTS_DIR, 'codeformer.pth')
RESTOREFORMER_MODEL_PATH = os.path.join(RESTOREFORMER_WEIGHTS_DIR, 'restoreformer.pth')
WAV2LIP_MODEL_PATH = os.path.join(WAV2LIP_WEIGHTS_DIR, 'wav2lip.pth')
WAV2LIP_GAN_MODEL_PATH = os.path.join(WAV2LIP_WEIGHTS_DIR, 'wav2lip_gan.pth')
VQFR1_MODEL_PATH = os.path.join(VQFR_WEIGHTS_DIR, 'VQFR_v1-33a1fac5.pth')
VQFR2_MODEL_PATH = os.path.join(VQFR_WEIGHTS_DIR, 'VQFR_v2.pth')

def __init__():
    perform_check()
    archs.__init__()

def download_from_drive(url, model_dir, progress, file_name):
    output_path = os.path.join(model_dir, file_name)
    try:
        gdown.download(url, output=output_path, quiet=(progress is False))
    except RuntimeError as e:
        print(f"Error occurred while downloading from drive: {e}")

def perform_check(bg_model_name='RealESRGAN_x2plus', restorer='CodeFormer', use_gan_version=False):
    REALESRGAN_MODEL_PATH = os.path.join(REALESRGAN_WEIGHTS_DIR, f'{bg_model_name}.pth')
    try:
        #------------------------------CHECK FOR TEMP DIR-------------------------------
        # Check if directory exists
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
            os.makedirs(NPY_FILES_DIR)
            os.makedirs(MEDIA_DIR)

        #------------------------------CHECK FOR OUTPUT DIR-----------------------------
        # Check if directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        #------------------------------CHECK FOR WEIGHTS--------------------------------
        # Check if directory exists
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)
            os.makedirs(MP_WEIGHTS_DIR)
            os.makedirs(GFPGAN_WEIGHTS_DIR)
            os.makedirs(CODEFORMERS_WEIGHTS_DIR)
            os.makedirs(WAV2LIP_WEIGHTS_DIR)

        # Download face landmark and face detector models
        if not os.path.exists(MP_LANDMARKER_MODEL_PATH):
            print("Downloading Face Landmarker model...")
            load_file_from_url(url=LANDMARKER_MODEL_URL, 
                               model_dir=MP_WEIGHTS_DIR,
                               progress=True, 
                               file_name='face_landmarker.task')
            
        if not os.path.exists(MP_DETECTOR_MODEL_PATH):
            print("Downloading Face Detector model...")
            load_file_from_url(url=DETECTOR_MODEL_URL, 
                               model_dir=MP_WEIGHTS_DIR,
                               progress=True, 
                               file_name='blaze_face_short_range.tflite')
        
        # Download restorer model
        if restorer == 'GFPGAN':
            if not os.path.exists(GFPGAN_MODEL_PATH):
                print("Downloading GFPGAN model...")
                load_file_from_url(url=GFPGAN_MODEL_URL, 
                                model_dir=GFPGAN_WEIGHTS_DIR,
                                progress=True,
                                file_name='GFPGANv1.4.pth')
        
        elif restorer == 'CodeFormer':
            if not os.path.exists(CODEFORMERS_MODEL_PATH):
                print("Downloading CodeFormer model...")
                load_file_from_url(url=CODEFORMERS_MODEL_URL,
                                model_dir=CODEFORMERS_WEIGHTS_DIR,
                                progress=True,
                                file_name='codeformer.pth')
                
        elif restorer == 'RestoreFormer':
            if not os.path.exists(RESTOREFORMER_MODEL_PATH):
                print("Downloading RestoreFormer model...")
                load_file_from_url(url=RESTOREFORMER_MODEL_URL,
                                model_dir=RESTOREFORMER_WEIGHTS_DIR,
                                progress=True,
                                file_name='restoreformer.pth')
                
        elif restorer == 'VQFR1':
            if not os.path.exists(VQFR1_MODEL_PATH):
                print("Downloading VQFR model...")
                load_file_from_url(url=VQFR1_MODEL_URL,
                                model_dir=VQFR_WEIGHTS_DIR,
                                progress=True,
                                file_name='VQFR_v1-33a1fac5.pth')
                
        elif restorer == 'VQFR2':
            if not os.path.exists(VQFR2_MODEL_PATH):
                print("Downloading VQFR model...")
                load_file_from_url(url=VQFR2_MODEL_URL,
                                model_dir=VQFR_WEIGHTS_DIR,
                                progress=True,
                                file_name='VQFR_v2.pth')
        
        # Download lip sync model
        if use_gan_version:
            if not os.path.exists(WAV2LIP_GAN_MODEL_PATH):
                print("Downloading Wav2Lip GAN model...")
                download_from_drive(url=WAV2LIP_GAN_MODEL_URL[1],
                                model_dir=WAV2LIP_WEIGHTS_DIR,
                                progress=True,
                                file_name='wav2lip_gan.pth')

        else:
            if not os.path.exists(WAV2LIP_MODEL_PATH):
                print("Downloading Wav2Lip model...")
                download_from_drive(url=WAV2LIP_MODEL_URL[1],
                                model_dir=WAV2LIP_WEIGHTS_DIR,
                                progress=True,
                                file_name='wav2lip.pth')
        
        # Download Real-ESRGAN model
        if not os.path.exists(REALESRGAN_MODEL_PATH):
            print(f"Downloading Real-ESRGAN model: {REALESRGAN_MODEL_PATH}...")
            load_file_from_url(url=REAL_ESRGAN_MODEL_URL[bg_model_name],
                                model_dir=REALESRGAN_WEIGHTS_DIR,
                                progress=True,
                                file_name=f'{bg_model_name}.pth')
    
    except OSError as e:
        print(f"OS Error occurred: {e}")
    except Exception as e:
        print(f"Unexpected Error occurred while performing file_check: {e}")

def get_file_type(filename):
    image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
    video_extensions = ["mp4", "mov", "avi", "mkv", "flv"]
    audio_extensions = ["mp3", "wav", "flac", "ogg", "m4a"]

    extension = filename.split('.')[-1].lower()

    if extension in image_extensions:
        return "image", extension
    elif extension in video_extensions:
        return "video", extension
    elif extension in audio_extensions:
        return "audio", extension
    else:
        return "unknown", extension
    
if __name__ == "__main__":
    perform_check()