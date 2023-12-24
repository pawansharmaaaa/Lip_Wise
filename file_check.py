import os
import uuid
from basicsr.utils.download_util import load_file_from_url

LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
DETECTOR_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'
GFPGAN_MODEL_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
WAV2LIP_MODEL_URL = 'https://drive.google.com/u/0/uc?id=1paYmN1KAZ2oPQPV-XauCoRUorhkOt0s2&export=download'
WAV2LIP_GAN_MODEL_URL = 'https://drive.google.com/u/0/uc?id=1WpqCULKQQcaCNf827h1qgjMHZENYHk-_&export=download'

CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_DIR = os.path.join(CURRENT_FILE_DIRECTORY, 'weights')
MP_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'mp')
GFPGAN_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'gfpgan')
WAV2LIP_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'wav2lip')
NPY_FILES_DIR = os.path.join(CURRENT_FILE_DIRECTORY, 'temp', 'npy_files')

print(CURRENT_FILE_DIRECTORY)

def perform_check():
    try:
        #------------------------------CHECK FOR TEMP DIR-------------------------------
        # Check if directory exists
        if not os.path.exists(os.path.join(CURRENT_FILE_DIRECTORY,'temp')):
            os.makedirs(os.path.join(CURRENT_FILE_DIRECTORY, 'temp'))
            os.makedirs(os.path.join(CURRENT_FILE_DIRECTORY, 'temp', 'npy_files'))
            os.makedirs(os.path.join(CURRENT_FILE_DIRECTORY, 'temp', 'media'))
            
        #------------------------------CHECK FOR WEIGHTS--------------------------------
        # Check if directory exists
        if not os.path.exists(os.path.join(CURRENT_FILE_DIRECTORY,'weights')):
            os.makedirs(os.path.join(CURRENT_FILE_DIRECTORY, 'weights'))
            os.makedirs(os.path.join(CURRENT_FILE_DIRECTORY, 'weights', 'mp'))
            os.makedirs(os.path.join(CURRENT_FILE_DIRECTORY, 'weights', 'gfpgan'))
            os.makedirs(os.path.join(CURRENT_FILE_DIRECTORY, 'weights', 'wav2lip'))


        if not os.path.exists(os.path.join(MP_WEIGHTS_DIR, 'face_landmarker.task')):
            print("Downloading Face Landmarker model...")
            load_file_from_url(url=LANDMARKER_MODEL_URL, 
                               model_dir=os.path.join(MP_WEIGHTS_DIR, 'face_landmarker.task'), 
                               progress=True, 
                               file_name=None)
            
        if not os.path.exists('blaze_face_short_range.tflite'):
            print("Downloading Face Detector model...")
            load_file_from_url(url=DETECTOR_MODEL_URL, 
                               model_dir=os.path.join(MP_WEIGHTS_DIR,'blaze_face_short_range.tflite'),
                               progress=True, 
                               file_name=None)
            
        if not os.path.exists('GFPGANv1.4.pth'):
            print("Downloading GFPGAN model...")
            load_file_from_url(url=GFPGAN_MODEL_URL, 
                               model_dir=os.path.join(GFPGAN_WEIGHTS_DIR, 'GFPGANv1.4.pth'),
                               progress=True, 
                               file_name=None)
            
        if not os.path.exists('wav2lip.pth'):
            print("Downloading Wav2Lip model...")
            load_file_from_url(url=WAV2LIP_MODEL_URL, 
                               model_dir=os.path.join(WAV2LIP_WEIGHTS_DIR, 'wav2lip.pth'),
                               progress=True,
                               file_name=None)
            
        if not os.path.exists('wav2lip_gan.pth'):
            print("Downloading Wav2Lip GAN model...")
            load_file_from_url(url=WAV2LIP_GAN_MODEL_URL,
                               model_dir=os.path.join(WAV2LIP_WEIGHTS_DIR, 'wav2lip_gan.pth'),
                               progress=True,
                               file_name=None)
    
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