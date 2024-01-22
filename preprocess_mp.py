# This file is a part of https://github.com/pawansharmaaaa/Lip_Wise/ repository.

import cv2
import math
import os, sys
import file_check

import mediapipe as mp
import numpy as np

class model_processor:

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

    def __init__(self, padding=0):
        self.padding = padding
        self.npy_directory = file_check.NPY_FILES_DIR
        self.weights_directory = file_check.WEIGHTS_DIR

        self.detector_model_path = os.path.join(file_check.MP_WEIGHTS_DIR, 'blaze_face_short_range.tflite')
        self.landmarker_model_path = os.path.join(file_check.MP_WEIGHTS_DIR, "face_landmarker.task")
        self.gen_face_route_index()

    def detect_for_image(self, frame):

        # frame = cv2.imread(image_path)
        # Convert frame to RGB and convert to MediaPipe image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        norm_pad_x = self.padding / width
        norm_pad_y = self.padding / height
        
        # Initialize mediapipe
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

        # Create a face detector instance with the image mode:
        options_det = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=self.detector_model_path),
            min_detection_confidence=0.5,
            running_mode=VisionRunningMode.IMAGE)


        options_lan = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.landmarker_model_path),
            min_face_detection_confidence=0.5,
            running_mode=VisionRunningMode.IMAGE)
        
        image_landmarks = np.zeros((1, 486, 2)).astype(np.float64)
        
        with FaceLandmarker.create_from_options(options_lan) as landmarker,FaceDetector.create_from_options(options_det) as detector:
            # Run face detector and face landmark models in IMAGE mode
            result_landmarker = landmarker.detect(mp_frame)
            result_detection = detector.detect(mp_frame)
        
            # Get data ready to be saved
            if len(result_detection.detections) > 0 and len(result_landmarker.face_landmarks) > 0:
                # Get Keypoints
                kp = result_detection.detections[0].keypoints
                kp_np = np.array([[k.x, k.y] for k in kp]).astype(np.float64)

                # Get landmarks
                landmarks_np = np.array([[i.x, i.y] for i in result_landmarker.face_landmarks[0]]).astype(np.float64)

                # Get bounding box
                # x-coordinates are at even indices and y-coordinates are at odd indices
                x_coordinates = landmarks_np[:, 0]
                y_coordinates = landmarks_np[:, 1]

                # Top-most point has the smallest y-coordinate
                y_min = landmarks_np[np.argmin(y_coordinates)] - norm_pad_y

                # Bottom-most point has the largest y-coordinate
                y_max = landmarks_np[np.argmax(y_coordinates)] + norm_pad_y

                # Left-most point has the smallest x-coordinate
                x_min = landmarks_np[np.argmin(x_coordinates)] - norm_pad_x

                # Right-most point has the largest x-coordinate
                x_max = landmarks_np[np.argmax(x_coordinates)] + norm_pad_x

                bbox_np = np.array([[x_min[0], y_min[1]], [x_max[0], y_max[1]]]).astype(np.float64)

                # Concatenate landmarks, bbox and keypoints. This is the data that will be saved.
                data = np.vstack((landmarks_np, bbox_np, kp_np)).astype(np.float64)
            else:
                data = np.zeros((486,2)).astype(np.float64)
            
            image_landmarks[0] = data
            np.save(os.path.join(self.npy_directory,'image_landmarks.npy'), image_landmarks)

    def detect_for_video(self, video_path):
        try:
            video = cv2.VideoCapture(video_path)
        except Exception as e:
            print(f"Exception occurred while trying to open video file. Exceptin: {e}")
            exit(1)

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize mediapipe
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

        # Create a face detector instance with the image mode:
        options_det = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=self.detector_model_path),
            min_detection_confidence=0.5,
            running_mode=VisionRunningMode.IMAGE)


        options_lan = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.landmarker_model_path),
            min_face_detection_confidence=0.5,
            running_mode=VisionRunningMode.IMAGE)

        frame_no = 0
        no_face_index = []
        video_landmarks = np.zeros((frame_count, 486, 2)).astype(np.float64)
        norm_pad_x = self.padding / width
        norm_pad_y = self.padding / height

        with FaceLandmarker.create_from_options(options_lan) as landmarker,FaceDetector.create_from_options(options_det) as detector:
            while video.isOpened():

                ret, frame = video.read()
                
                if not ret:
                    break

                # Convert frame to RGB and convert to MediaPipe image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                # Run face detector and face landmark models in IMAGE mode
                result_landmarker = landmarker.detect(mp_frame)
                result_detection = detector.detect(mp_frame)
                
                if len(result_detection.detections) > 0 and len(result_landmarker.face_landmarks) > 0:
                    # Get Keypoints
                    kp = result_detection.detections[0].keypoints
                    kp_np = np.array([[k.x, k.y] for k in kp]).astype(np.float64)

                    # Get landmarks
                    landmarks_np = np.array([[i.x, i.y] for i in result_landmarker.face_landmarks[0]]).astype(np.float64)

                    # Get bounding box
                    # x-coordinates are at even indices and y-coordinates are at odd indices
                    x_coordinates = landmarks_np[:, 0]
                    y_coordinates = landmarks_np[:, 1]

                    # Top-most point has the smallest y-coordinate
                    y_min = landmarks_np[np.argmin(y_coordinates)] - norm_pad_y

                    # Bottom-most point has the largest y-coordinate
                    y_max = landmarks_np[np.argmax(y_coordinates)] + norm_pad_y

                    # Left-most point has the smallest x-coordinate
                    x_min = landmarks_np[np.argmin(x_coordinates)] - norm_pad_x

                    # Right-most point has the largest x-coordinate
                    x_max = landmarks_np[np.argmax(x_coordinates)] + norm_pad_x

                    bbox_np = np.array([[x_min[0], y_min[1]], [x_max[0], y_max[1]]]).astype(np.float64)

                    # Concatenate landmarks, bbox and keypoints. This is the data that will be saved.
                    data = np.vstack((landmarks_np, bbox_np, kp_np)).astype(np.float64)
                else:
                    data = np.zeros((486,2)).astype(np.float64)
                    no_face_index.append(frame_no)

                # Append data
                video_landmarks[frame_no] = data
            
                # Increment frame number
                frame_no += 1
            
            # Save video landmarks
            np.save(os.path.join(self.npy_directory,'video_landmarks.npy'), video_landmarks)
            np.save(os.path.join(self.npy_directory,'no_face_index.npy'), np.array(no_face_index))

            # Release video
            video.release()

    def gen_face_route_index(self):
            """
            Generates and saves the face route index as an npy array.

            The face route index is obtained by sorting the face oval indices in a specific order.

            Returns:
                None
            """
            # Load face mesh model
            mp_face_mesh = mp.solutions.face_mesh
            face_oval = mp_face_mesh.FACEMESH_FACE_OVAL

            index = np.array(list(face_oval))

            # Sort the array
            i = 0
            while i < len(index) - 1:
                # Find the index of the row where the first element is the same as the second element of the current row
                next_index = np.where(index[:, 0] == index[i, 1])[0]
                if next_index.size>0 and next_index[0] != i + 1:
                    # Swap rows
                    index[[i + 1, next_index[0]]] = index[[next_index[0], i + 1]]
                i += 1

            # Save index as npy array
            os.makedirs(self.npy_directory, exist_ok=True)
            np.save(os.path.join(self.npy_directory, 'face_route_index.npy'), index)

    def align_3d(self, frame):
        """
        Aligns the face in the image in 3D.

        Args:
            frame: The image from which the face is to be aligned.

        Returns:
            The aligned face.
        """
        # Load the model (adjust the path as needed)
        face_aligner = mp.tasks.vision.FaceAligner.create_from_model_path(self.landmarker_model_path)

        # Load your image (replace with your image path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Align the face(s) in the image
        aligned_image = face_aligner.align(mp_frame)

        if aligned_image:
            # Convert the MediaPipe image to OpenCV image
            aligned_image = cv2.cvtColor(aligned_image.numpy_view(), cv2.COLOR_RGB2BGR)
            return aligned_image
        else:
            # No face detected in the image
            print("3D Alignment failed. No face detected in the image.")
            sys.exit(1)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
class FaceHelpers:

    def __init__(self, image_mode=False, max_batch_size=16):

        self.max_batch_size = max_batch_size
        self.npy_directory = file_check.NPY_FILES_DIR
        self.weights_directory = file_check.WEIGHTS_DIR

        self.video_landmarks_path = os.path.join(self.npy_directory,'video_landmarks.npy')
        self.image_landmarks_path = os.path.join(self.npy_directory,'image_landmarks.npy')
        if image_mode:
            try:
                self.landmarks_all = np.load(self.image_landmarks_path)
            except FileNotFoundError as e:
                print("Image landmarks were not saved. Please report this issue.")
                exit(1)
        else:
            try:
                self.landmarks_all = np.load(self.video_landmarks_path)
            except FileNotFoundError as e:
                print("Video landmarks were not saved. Please report this issue.")
                exit(1)

    def gen_face_mask(self, img, frame_no=0):
        """
        Generates the face mask using face oval indices from face oval landmarks.

        Args:
            img: Image from which the face mask is to be generated.
            frame_no: The frame number of the image.
        
        Returns:
            The face mask in bool format to be fed in the paste back function or extract_face function.
        """
        if not os.path.exists(os.path.join(self.npy_directory, 'face_route_index.npy')):
            processor = model_processor()
            processor.gen_face_route_index()
        try:
            index = np.load(os.path.join(self.npy_directory, 'face_route_index.npy'))
        except FileNotFoundError as e:
            print("Face route index was not read. Please rerun the inference.")
            exit(1)

        coords = list()
        for source_idx, target_idx in index:
            source = self.landmarks_all[frame_no][source_idx]
            target = self.landmarks_all[frame_no][target_idx]
            coords.append([source[0], source[1]])
            coords.append([target[0], target[1]])

        coords = np.array(coords)
        coords = (coords*[img.shape[1], img.shape[0]]).astype(np.int32)

        mask = np.zeros((img.shape[0], img.shape[1]))
        mask = cv2.fillConvexPoly(mask, coords, 1)
        mask = mask.astype(bool)

        original_mask = mask

        return original_mask

    def extract_face(self, original_img, frame_no=0):
        """
        Extracts the face from the image (Image with only face and black background).

        Args:
            img: Image from which the face is to be extracted.

        Returns:
            Only The face.
        """
        original_mask = self.gen_face_mask(original_img, frame_no)
        extracted_face = np.zeros_like(original_img)
        extracted_face[original_mask] = original_img[original_mask]

        return extracted_face, original_mask
    

    def findEuclideanDistance(self, source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    #this function is inspired from the deepface repository: https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
    def alignment_procedure(self, extracted_face, frame_no=0):

        height, width = extracted_face.shape[:2]

        left_eye = self.landmarks_all[frame_no][480] *  [width, height]# Left eye index is 480, and also unzipping
        right_eye = self.landmarks_all[frame_no][481] * [width, height] # Right eye index is 481, and also unzipping
        #this function aligns given face in img based on left and right eye coordinates

        #left eye is the eye appearing on the left (right eye of the person)
        #left top point is (0, 0)

        left_eye_x = left_eye[0]
        left_eye_y = left_eye[1]
        right_eye_x = right_eye[0]
        right_eye_y = right_eye[1]

        #-----------------------
        #decide the image is inverse

        center_eyes = (int((left_eye_x + right_eye_x) / 2), int((left_eye_y + right_eye_y) / 2))

        center = (extracted_face.shape[1] / 2, extracted_face.shape[0] / 2)

        output_size = (extracted_face.shape[1], extracted_face.shape[0])
        

        #-----------------------
        #find rotation direction

        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 #rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 #rotate inverse direction of clock

        #-----------------------
        #find length of triangle edges

        a = self.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
        b = self.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
        c = self.findEuclideanDistance(np.array(right_eye), np.array(left_eye))

        #-----------------------

        #apply cosine rule

        if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

            cos_a = (b*b + c*c - a*a)/(2*b*c)
            
            #PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0], floating point errors would produce cases violating this
            #In fact, we did come across a case where cos_a took the value 1.0000000169176173, which lead to a NaN from the following np.arccos step
            
            cos_a = np.clip(cos_a, -1.0, 1.0)
            angle = np.arccos(cos_a) #angle in radian
            angle = (angle * 180) / math.pi #radian to degree

            #-----------------------
            #rotate base image

            if direction == -1:
                angle = 90 - angle

            try:
                # Get the rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, direction * angle, 1)

                # Perform the affine transformation to rotate the image
                aligned_face = cv2.warpAffine(extracted_face, rotation_matrix, output_size)
            except Exception as e:
                print(f"Error aligning face at frame no: {frame_no}, Saving the frame for manual inspection.")
                os.makedirs(os.path.join(file_check.CURRENT_FILE_DIRECTORY, 'error_frames'), exist_ok=True)
                cv2.imwrite(os.path.join(file_check.CURRENT_FILE_DIRECTORY, 'error_frames', f'frame_{frame_no}.jpg'), extracted_face)
                exit(1)

        return aligned_face, rotation_matrix  #return img and inverse afiine matrix anyway
    
    def crop_extracted_face(self, aligned_face, rotation_matrix, frame_no=0):
        """
        Crops the face from the image (Image with only face and black background).

        Args:
            face: Image from which the face is to be cropped.

        Returns:
            Only The face.
        """
        old_bbox = self.landmarks_all[frame_no][478:480].copy()
        old_landmarks = self.landmarks_all[frame_no][480:].copy()

        # Unzip the bbox and landmarks
        old_bbox = old_bbox * [aligned_face.shape[1], aligned_face.shape[0]]
        old_landmarks = old_landmarks * [aligned_face.shape[1], aligned_face.shape[0]]

        # Get new center from rotation_matrix
        new_center = (rotation_matrix[:,2:])
        new_center = np.reshape(new_center, (1, 2))

        # New_landmarks
        rotated_landmarks = np.dot(old_landmarks, rotation_matrix[:,:2].T) + new_center

        # Calculate Distances for bbox
        d1 = abs(old_landmarks[4][0] - old_bbox[0][0])
        d2 = abs(old_landmarks[5][0] - old_bbox[1][0])
        d3 = abs(max(old_landmarks[0][1], old_landmarks[1][1]) - old_bbox[0][1])
        d4 = abs(old_landmarks[3][1] - old_bbox[1][1])

        xmin = rotated_landmarks[4][0] - d1
        xmax = rotated_landmarks[5][0] + d2
        ymin = rotated_landmarks[0][1] - d3
        ymax = rotated_landmarks[3][1] + d4

        # New_bbox
        bbox = np.array([[xmin, ymin], [xmax, ymax]]).astype(np.int32)

        # Crop face
        try:
            cropped_face = aligned_face[bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0]]
        except IndexError as e:
            print(f"Failed to crop face: {e}")
            print(f"Saving the frame for manual inspection.")
            os.makedirs(os.path.join(file_check.CURRENT_FILE_DIRECTORY, 'error_frames'), exist_ok=True)
            cv2.imwrite(os.path.join(file_check.CURRENT_FILE_DIRECTORY, 'error_frames', f'frame_crop{frame_no}.jpg'), aligned_face)
            exit(1)

        return cropped_face, bbox
    
        
    def gen_data_image_mode(self, cropped_face, mel_chunks):
        """
        Generates data for inference in image mode.
        Batches the data to be fed into the model.
        Batch of image includes several images of shape (96, 96, 6) stacked together.
        These images contain the half face and the full face.

        Args:
            cropped_face: The cropped face obtained from the crop_extracted_face function.
            mel_chunks: The mel chunks obtained from the audio.
        
        Returns:
            A batch of images of shape (96, 96, 6) and mel chunks.
        """
        frame_batch = []
        mel_batch = []

        # Resize face for wav2lip
        try:
            cropped_face = cv2.resize(cropped_face, (96, 96), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Failed to resize face: {e}")
            exit(1)

        # Generate data for inference
        for mel_chunk in mel_chunks:
            frame_batch.append(cropped_face)
            mel_batch.append(mel_chunk)

            if len(frame_batch) >= self.max_batch_size:
                frame_batch, mel_batch = np.asarray(frame_batch), np.asarray(mel_batch)

                img_masked = frame_batch.copy()
                img_masked[:, 96//2:] = 0

                frame_batch = np.concatenate((img_masked, frame_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield frame_batch, mel_batch
                frame_batch = []
                mel_batch = []

        if len(frame_batch) > 0:
            frame_batch, mel_batch = np.asarray(frame_batch), np.asarray(mel_batch)

            img_masked = frame_batch.copy()
            img_masked[:, 96//2:] = 0

            frame_batch = np.concatenate((img_masked, frame_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield frame_batch, mel_batch

    def paste_back_black_bg(self, processed_face, bbox, full_frame):
        processed_ready = np.zeros_like(full_frame)
        try:
            processed_ready[bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0]] = processed_face
        except IndexError as e:
            print(f"Failed to paste face back onto full frame: {e}")
            print(f"Saving the frame for manual inspection.")
            os.makedirs(os.path.join(file_check.CURRENT_FILE_DIRECTORY, 'error_frames'), exist_ok=True)
            cv2.imwrite(os.path.join(file_check.CURRENT_FILE_DIRECTORY, 'error_frames', f'frame_paste.jpg'), processed_face)
            exit(1)

        return processed_ready
    
    def unwarp_align(self, processed_ready, rotation_matrix):
        """
        Unwarps and unaligns the processed face.

        Args:
            processed_face: The processed face.
            rotation_matrix: The rotation matrix obtained from the alignment procedure.
            bbox: The bounding box of the face.
        
        Returns:
            The unwarped and unaligned face.
        """
        # Unwarp and unalign
        # print("Unwarping and unaligning face...")
        try:
            ready_to_paste = cv2.warpAffine(processed_ready, rotation_matrix, (processed_ready.shape[1], processed_ready.shape[0]), flags=cv2.WARP_INVERSE_MAP)
        except Exception as e:
            print(f"Failed to unwarp and unalign face: {e}")
            print(f"Saving the frame for manual inspection.")
            os.makedirs(os.path.join(file_check.CURRENT_FILE_DIRECTORY, 'error_frames'), exist_ok=True)
            cv2.imwrite(os.path.join(file_check.CURRENT_FILE_DIRECTORY, 'error_frames', f'frame_unwarp.jpg'), processed_ready)
            exit(1)
        return ready_to_paste


    def paste_back(self, ready_to_paste, background, original_mask):
        """
        Pastes the face back on the background.

        Args:
            face: Full image with the face.
            background: The background on which the face is to be pasted.

        Returns:
            The background with the face pasted on it.
        """
        # print("Pasting face back...")
        try:
            background[original_mask] = ready_to_paste[original_mask]
        except IndexError as e:
            print(f"Failed to paste face back onto background: {e}")
            print(f"Saving the frame for manual inspection.")
            os.makedirs(os.path.join(file_check.CURRENT_FILE_DIRECTORY, 'error_frames'), exist_ok=True)
            cv2.imwrite(os.path.join(file_check.CURRENT_FILE_DIRECTORY, 'error_frames', f'frame_paste_back.jpg'), ready_to_paste)
            exit(1)
        return background