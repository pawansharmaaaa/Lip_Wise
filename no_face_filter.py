import cv2
import face_recognition
import gc
import tqdm

# Function to detect face in a frame
def detect_face(frames):
    batch_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)
    return batch_face_locations

# Open the video file for reading
def filter_face(input_path, output_path):

    video_capture = cv2.VideoCapture(input_path)

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writers for face.mp4
    face_writer = cv2.VideoWriter(input_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    batch_size = 16
    frames = []
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        # Detect faces in the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if len(frames) == batch_size:
            # Detect faces in the frame
            batch_faces = detect_face(frames)

            for frame_no, face_locations in enumerate(tqdm.tqdm(batch_faces, total=batch_size)):
                if len(face_locations) == 1:
                    # Write the frame to input_video.mp4 if a face is detected
                    pos_frame = cv2.cvtColor(frames[frame_no], cv2.COLOR_RGB2BGR)
                    face_writer.write(pos_frame)
            frames = []

    # Release video capture, writers and clear garbage
    video_capture.release()
    face_writer.release()
    cv2.destroyAllWindows()
    gc.collect()

    return output_path