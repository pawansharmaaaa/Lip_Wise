import os
import uuid
from basicsr.utils.download_util import load_file_from_url

LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
DETECTOR_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'
GFPGAN_MODEL_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
WAV2LIP_MODEL_URL = 'https://drive.google.com/uc?export=download&id=1paYmN1KAZ2oPQPV-XauCoRUorhkOt0s2'
WAV2LIP_GAN_MODEL_URL = 'https://drive.google.com/uc?export=download&id=1WpqCULKQQcaCNf827h1qgjMHZENYHk-_'

CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_DIR = os.path.join(CURRENT_FILE_DIRECTORY, 'weights')
MP_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'mp')
GFPGAN_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'gfpgan')
WAV2LIP_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'wav2lip')

TEMP_DIR = os.path.join(CURRENT_FILE_DIRECTORY, 'temp')
NPY_FILES_DIR = os.path.join(TEMP_DIR, 'npy_files')
MEDIA_DIR = os.path.join(TEMP_DIR, 'media')

OUTPUT_DIR = os.path.join(CURRENT_FILE_DIRECTORY, 'output')

MP_LANDMARKER_MODEL_PATH = os.path.join(MP_WEIGHTS_DIR, 'face_landmarker.task')
MP_DETECTOR_MODEL_PATH = os.path.join(MP_WEIGHTS_DIR, 'blaze_face_short_range.tflite')
GFPGAN_MODEL_PATH = os.path.join(GFPGAN_WEIGHTS_DIR, 'GFPGANv1.4.pth')
WAV2LIP_MODEL_PATH = os.path.join(WAV2LIP_WEIGHTS_DIR, 'wav2lip.pth')
WAV2LIP_GAN_MODEL_PATH = os.path.join(WAV2LIP_WEIGHTS_DIR, 'wav2lip_gan.pth')

def perform_check():
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
            os.makedirs(WAV2LIP_WEIGHTS_DIR)


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
            
        if not os.path.exists(GFPGAN_MODEL_PATH):
            print("Downloading GFPGAN model...")
            load_file_from_url(url=GFPGAN_MODEL_URL, 
                               model_dir=GFPGAN_WEIGHTS_DIR,
                               progress=True,
                               file_name='GFPGANv1.4.pth')
            
        if not os.path.exists(WAV2LIP_MODEL_PATH):
            print("Downloading Wav2Lip model...")
            load_file_from_url(url=WAV2LIP_MODEL_URL, 
                               model_dir=WAV2LIP_WEIGHTS_DIR,
                               progress=True,
                               file_name='wav2lip.pth')
            
        if not os.path.exists(WAV2LIP_GAN_MODEL_PATH):
            print("Downloading Wav2Lip GAN model...")
            load_file_from_url(url=WAV2LIP_GAN_MODEL_URL,
                               model_dir=WAV2LIP_WEIGHTS_DIR,
                               progress=True,
                               file_name='wav2lip_gan.pth')
    
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