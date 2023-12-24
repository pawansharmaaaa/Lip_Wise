import cv2
import mediapipe as mp
import numpy as np

"""
INDEXES:
    0-477: Landmarks:
    478-479: Bounding box:
        478: Top left corner
        479: height, width
    480-485: Keypoints:
        480: Left eye
        481: Right eye
        482: Nose
        483: Mouth left
        484: Mouth right
        485: Mouth center
"""
#add print statements to see what is going on, add file checkers, add file savers

def preprocess_video(video_path):

    video = cv2.VideoCapture(video_path)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Frames to process: {frame_count}")

    # Initialize mediapipe
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

    # Create a face detector instance with the image mode:
    options_det = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path='E:\\Lip_Wise_GFPGAN\\_tests\\mp\\weights\\blaze_face_short_range.tflite'),
        min_detection_confidence=0.5,
        running_mode=VisionRunningMode.IMAGE)


    options_lan = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="E:\\Lip_Wise_GFPGAN\\_tests\\mp\\weights\\face_landmarker.task"),
        min_face_detection_confidence=0.5,
        running_mode=VisionRunningMode.IMAGE)

    frame_no = 0
    no_face_index = []
    video_landmarks = np.zeros((frame_count, 486, 2)).astype(np.float64)

    with FaceLandmarker.create_from_options(options_lan) as landmarker,FaceDetector.create_from_options(options_det) as detector:
        while video.isOpened():

            ret, frame = video.read()
            
            if not ret:
                break
            
            # Get frame timestamp
            timestamp = int(video.get(cv2.CAP_PROP_POS_MSEC))

            # Convert frame to RGB and convert to MediaPipe image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Run face detector and face landmark models in VIDEO mode
            # result_landmarker = landmarker.detect_for_video(mp_frame, timestamp)
            # result_detection = detector.detect_for_video(mp_frame, timestamp)

            # Run face detector and face landmark models in IMAGE mode
            result_landmarker = landmarker.detect(mp_frame)
            result_detection = detector.detect(mp_frame)

            # Get data ready to be saved
            print(f"Frame {frame_no} processing")
            print(f"Face Detected: {len(result_landmarker.face_landmarks) > 0}")
            if len(result_detection.detections) > 0 and len(result_landmarker.face_landmarks) > 0:
                # Get bounding box
                bbox = result_detection.detections[0].bounding_box
                bbox_np = (np.array([bbox.origin_x, bbox.origin_y, bbox.width, bbox.height]).reshape(2, 2) / [width, height]).astype(np.float64)

                # Get Keypoints
                kp = result_detection.detections[0].keypoints
                kp_np = np.array([[k.x, k.y] for k in kp]).astype(np.float64)

                # Get landmarks
                landmarks_np = np.array([[i.x, i.y] for i in result_landmarker.face_landmarks[0]]).astype(np.float64)

                # Concatenate landmarks, bbox and keypoints. This is the data that will be saved.
                data = np.vstack((landmarks_np, bbox_np, kp_np)).astype(np.float64)
            else:
                data = np.zeros((486,2)).astype(np.float64)
                no_face_index.append(frame_no)

            # Append data
            print(f"Frame {frame_no} processed")
            video_landmarks[frame_no] = data
        
            # Increment frame number
            frame_no += 1
        # Save video landmarks
        np.save('E:\\Lip_Wise_GFPGAN\\_testData\\Outputs\\video_landmarks.npy', video_landmarks)
        np.savetxt('E:\\Lip_Wise_GFPGAN\\_testData\\Outputs\\no_face_index_1.txt', np.array(no_face_index), fmt='%d')

        # Release video
        video.release()