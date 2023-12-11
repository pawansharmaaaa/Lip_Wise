import argparse
import numpy as np
from models import YuNet
import cv2 as cv


# Check OpenCV version
assert cv.__version__ >= "4.8.0", \
    "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(
    description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set input to a certain image, omit if using camera.')
parser.add_argument('--model', '-m', type=str, default='./checkpoints/face_detection_yunet_2023mar.onnx',
                    help="Usage: Set model type, defaults to 'face_detection_yunet_2023mar.onnx'.")
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
        0: (default) OpenCV implementation + CPU,
        1: CUDA + GPU (CUDA),
        2: CUDA + GPU (CUDA FP16),
        3: TIM-VX + NPU,
        4: CANN + NPU
    ''')
parser.add_argument('--conf_threshold', type=float, default=0.9,
                    help='Usage: Set the minimum needed confidence for the model to identify a face, defauts to 0.9. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3,
                    help='Usage: Suppress bounding boxes of iou >= nms_threshold. Default = 0.3.')
parser.add_argument('--top_k', type=int, default=5000,
                    help='Usage: Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()


def setup_model(video_capture):

    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Get video properties
    fps = video_capture.get(cv.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    print(f"frame_width: {frame_width}")
    frame_height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"frame_height: {frame_height}")

    # Instantiate YuNet
    model = YuNet(modelPath=args.model,
                  inputSize=[1920, 1080],
                  confThreshold=args.conf_threshold,
                  nmsThreshold=args.nms_threshold,
                  topK=args.top_k,
                  backendId=backend_id,
                  targetId=target_id)

    # Set model input size based on video frame size
    # model.setInputSize([frame_width, frame_height])

    return model, fps, frame_width, frame_height


def process_video(input_path, output_path):
    try:

        # Capture video
        video_capture = cv.VideoCapture(input_path)

        # Setup model and video writer
        model, fps, frame_width, frame_height = setup_model(video_capture)
        no_of_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))

        face_writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Process frames
        print('Processing video...')
        print('Number of frames: ', no_of_frames)
        count = 1
        batch_size = 10
        frames = []
        original_frames = []

        while True:
            # Read frame
            has_frame, frame = video_capture.read()
            if not has_frame:
                break

            # Add frame to batch
            image = frame.copy()
            original_frames.append(frame)
            frames.append(image)

            # Process batch
            if len(frames) == batch_size:
                print(f'Processing batch {count} of {no_of_frames // batch_size}')
                batch = np.stack(frames)
                faces = [model.infer(f) for f in batch]

                # Write frames with faces
                for i, face in enumerate(faces):
                    if len(face) >= 1:
                        # pos_frame = cv.cvtColor(batch[i], cv.COLOR_RGB2BGR)
                        face_writer.write(original_frames[i])
                        # cv.imshow('DEMO', original_frames[i])
                frames = []
                original_frames = []
                count += 1


        # Process remaining frames
        if frames:
            batch = np.stack(frames)
            faces = [model.infer(f) for f in batch]
            for i, face in enumerate(faces):
                if len(face) >= 1:
                    pos_frame = cv.cvtColor(batch[i], cv.COLOR_RGB2BGR)
                    face_writer.write(pos_frame)

        # Release resources
        face_writer.release()
        cv.destroyAllWindows()
        video_capture.release()
        print('Video processing completed successfully!')

    except Exception as e:
        print(f'Error occurred: {e}')
        cv.destroyAllWindows()
        video_capture.release()


if __name__ == '__main__':
    input_path = 'test.mp4'
    output_path = 'out.avi'
    process_video(input_path, output_path)