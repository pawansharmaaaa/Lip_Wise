import os
import cv2 as cv
import numpy as np
from contextlib import contextmanager

from models.yunet import YuNet

#------------------------- Available Backends------------------------------

backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU]
]

#------------------------- Configuration Variables-------------------------

class Config:
    def __init__(self, resize_factor, conf_threshold, nms_threshold, top_k, backend):
        self.resize_factor = resize_factor
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.backend = backend

# Initialize the configuration
def init_config(resize_factor, conf_threshold, nms_threshold, top_k, backend):
    config = Config(resize_factor, conf_threshold, nms_threshold, top_k, backend)
    return config

#------------------------- Helper Functions -------------------------------
'''
-----------------------------------------------------------------------------
|                                                                           |
|                       Write check functions here                          |
|                                                                           |
-----------------------------------------------------------------------------
'''
def trim_structured_zeros(arr):
    # Reverse the array
    reversed_arr = arr[::-1]
    
    # Find the index of the last non-zero element
    last_non_zero_index = next((i for i, x in enumerate(reversed_arr) if np.any(x['bbox'])), len(reversed_arr))
    
    # Reverse the array again and return the slice up to the last non-zero element
    return reversed_arr[::-1][:len(reversed_arr) - last_non_zero_index]

def setup_model(video_capture, model_path):
    
    config = init_config(0.5, 0.9, 0.3, 5000, 0)

    backend_id = backend_target_pairs[config.backend][0]
    target_id = backend_target_pairs[config.backend][1]

    # Get video properties
    fps = video_capture.get(cv.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    w = int(frame_width * config.resize_factor)
    h = int(frame_height * config.resize_factor)

    # Initialize model
    model = YuNet(modelPath=model_path,
                  inputSize=[w, h],
                  confThreshold=config.conf_threshold,
                  nmsThreshold=config.nms_threshold,
                  topK=config.top_k,
                  backendId=backend_id,
                  targetId=target_id)

    return model, frame_width, frame_height, fps, config.resize_factor

#------------------------- Main Functions ----------------------------
@contextmanager
def process_video(input_path, output_path, model_path, batch_size=16):
    try:

        # Capture video
        video_capture = cv.VideoCapture(input_path)

        # Setup model and video writer
        model, frame_width, frame_height, fps, resize_factor = setup_model(video_capture, model_path)
        face_writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        # Get number of frames
        no_of_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))

        batch_no = 0
        current_frame_index = 0
        filtered_frame_counter = 0

        # Create the structured arrays
        frames = np.zeros((batch_size, int(frame_height*resize_factor), int(frame_width*resize_factor), 3), dtype=np.uint8)
        original_frames = np.zeros((batch_size, frame_height, frame_width, 3), dtype=np.uint8)
        filtered_data = np.zeros((no_of_frames,), dtype=[('bbox', np.int32, (4,))])
        

        while True:
            # Read frame
            has_frame, frame = video_capture.read()
            if not has_frame:
                break

            # Add frame to batch
            image = frame.copy()
            image = cv.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
            original_frames[current_frame_index] = frame
            frames[current_frame_index] = image

            current_frame_index += 1

            # Process batch
            if current_frame_index == batch_size:
                current_frame_index = 0
                print(f'Processing batch {batch_no+1} of {no_of_frames // batch_size}')
                faces = [model.infer(f) for f in frames]

                # Write frames with faces to video
                for i, face in enumerate(faces):
                    if len(face) != 0:
                        bbox = np.array(face[0][0:4])
                        bbox = (bbox // resize_factor).astype(np.int32)
                        filtered_data[filtered_frame_counter]['bbox'] = bbox
                        filtered_frame_counter += 1
                        face_writer.write(original_frames[i])

                frames = frames * 0
                original_frames = original_frames * 0
                batch_no += 1
        
        filtered_data = trim_structured_zeros(filtered_data)
        np.save("E:\\Lip_Wise_GFPGAN\\_testData\\Outputs\\filtered_data.npy", filtered_data)

        # Release resources
        face_writer.release()
        cv.destroyAllWindows()
        video_capture.release()
        print('Video processing completed successfully!')

    except Exception as e:
        print(f'Error occurred: {e}')
        cv.destroyAllWindows()
        video_capture.release()

#------------------------- Main Function ----------------------------
if __name__ == '__main__':

    input_path = 'E:\\Lip_Wise_GFPGAN\\_testData\\Inputs\\small_test.mp4'
    output_path = 'E:\\Lip_Wise_GFPGAN\\_testData\\Outputs\\test_vid.mp4'
    model_path = "E:\\Lip_Wise_GFPGAN\\checkpoints\\face_detection_yunet_2023mar.onnx"

    process_video(input_path, output_path, model_path)

#------------------------- Model Downloader -------------------------
# from basicsr.utils.download_util import load_file_from_url
# if model_path.startswith('https://'):
#     model_path = load_file_from_url(url=model_path, model_dir=os.path.join(ROOT_DIR, 'gfpgan/weights'), progress=True, file_name=None)