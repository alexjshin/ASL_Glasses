import cv2
import os
import logging
import time
import numpy as np
import mediapipe as mp
import json
import multiprocessing
from tqdm import tqdm # for progress bars in Jupyter Notebook
from matplotlib import pyplot as plt
from functools import partial
import absl.logging
absl.logging.set_stderrthreshold('fatal')
logging.basicConfig(level=logging.ERROR)

# Suppress TensorFlow logging (which MediaPipe uses)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# logging.getLogger('tensorflow').setLevel(logging.FATAL)
# mp.solutions.drawing_utils._PRESENCE_THRESHOLD = 1.0  # Suppress drawing warnings

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_mp_frames(frame, holistic):
    # Opencv records in BGR while mediapipe supports RGB
    # We need to recolor frame to RGB to support MediaPipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  # Make Image non-writeable for performance
    results = holistic.process(image)              # process image and return object contain landmarks
    image.flags.writeable = True                   # Back to writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color back to BGR for OpenCV 

    return image, results

def draw_landmarks(image, results):
    # Draw face landmarks
    mp_drawing.draw_landmarks(image, 
                              results.face_landmarks, 
                              mp_holistic.FACEMESH_TESSELATION, # FACEMESH_CONTOURS could also be valid here
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(170,86,0), thickness=1, circle_radius=2),
                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    # Draw pose connections
    mp_drawing.draw_landmarks(image, 
                              results.pose_landmarks, 
                              mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, 
                              results.left_hand_landmarks, 
                              mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(), 
                              connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, 
                              results.right_hand_landmarks, 
                              mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(), 
                              connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

def extract_keypoints_comprehensive(results):
    # Pose Landmarks have 33 Keypoints
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    keypoints_vector = np.concatenate([pose, face, lh, rh])
    # print(keypoints_vector)
    return keypoints_vector

WLASL_PATH = "WLASL_DATA"
VIDEOS_PATH = os.path.join(WLASL_PATH, "videos")
JSON_PATH = os.path.join(WLASL_PATH, "WLASL_v0.3.json")
OUTPUT_PATH = "WLASL_Processed"

# Create Output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("WLASL folder contents:")
print(os.listdir(WLASL_PATH))

# If there's a JSON file, check its structure
json_path = os.path.join(WLASL_PATH, "WLASL_v0.3.json")
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"\nJSON file contains data for {len(data)} signs")
    print(f"Example of first sign: {data[0]['gloss']}")

def process_video_parallel(video_info, output_base_dir, max_frames=30):
    """Process a single video file and extract landmarks"""
    sign_name, video_idx, video_id = video_info
    
    # Create output directory for this video
    output_dir = os.path.join(output_base_dir, sign_name, str(video_idx))
    os.makedirs(output_dir, exist_ok=True)
    
    # Form video path
    video_path = os.path.join("WLASL_DATA", "videos", f"{video_id}.mp4")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return False
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling rate
        if frame_count <= max_frames:
            frame_indices = list(range(frame_count))
        else:
            frame_indices = np.linspace(0, frame_count-1, max_frames, dtype=int)
        
        # Initialize MediaPipe - create a new instance for each process
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Process Frames
            for frame_idx, i in enumerate(frame_indices):
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                success, frame = cap.read()
                if not success:
                    continue
                    
                # Process image
                _, results = process_mp_frames(frame, holistic)
                
                # Extract and save keypoints
                keypoints_vector = extract_keypoints_comprehensive(results)
                npy_path = os.path.join(output_dir, f"{frame_idx}.npy")
                np.save(npy_path, keypoints_vector)
        
        cap.release()
        return True
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False

def process_wlasl_dataset_parallel(max_signs=None, max_videos_per_sign=30, num_processes=None):
    """Process WLASL Dataset using parallel processing"""
    # If num processes isn't specified using all the available cores - 1
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)

    # Load WLASL metadata
    with open(os.path.join("WLASL_DATA", "WLASL_v0.3.json"), 'r') as f:
        wlasl_data = json.load(f)
    
    # Limit number of signs if specified
    if max_signs:
        wlasl_data = wlasl_data[:max_signs]

    # Create sign to ID mapping
    sign_map = {sign_data['gloss']: idx for idx, sign_data in enumerate(wlasl_data)}
    
    # Output directory
    output_path = "WLASL_Processed"
    os.makedirs(output_path, exist_ok=True)
    
    # Save sign map for later use
    with open(os.path.join(output_path, 'sign_map.json'), 'w') as f:
        json.dump(sign_map, f, indent=2)
    
    # Prepare list of videos to process
    videos_to_process = []
    for sign_idx, sign_data in enumerate(wlasl_data):
        sign_name = sign_data['gloss']
        
        # Create directory for this sign
        sign_dir = os.path.join(output_path, sign_name)
        os.makedirs(sign_dir, exist_ok=True)
        
        # Add videos for this sign to process list (limited by max_videos_per_sign)
        for video_idx, instance in enumerate(sign_data['instances'][:max_videos_per_sign]):
            video_id = instance['video_id']
            videos_to_process.append((sign_name, video_idx, video_id))
            
    # Create a partial function with fixed arguments
    process_func = partial(process_video_parallel, output_base_dir=output_path)
    
    # Process videos in parallel
    print(f"Processing {len(videos_to_process)} videos using {num_processes} processes...")
    
    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process videos and track progress with tqdm
        results = list(tqdm(
            pool.imap(process_func, videos_to_process),
            total=len(videos_to_process),
            desc="Processing videos"
        ))
    
    # Count processed videos per sign
    processed_counts = {}
    for sign_name, _, _ in videos_to_process:
        if sign_name not in processed_counts:
            processed_counts[sign_name] = 0
        processed_counts[sign_name] += 1
    
    # Create processed sign information
    processed_signs = [
        {
            'name': sign_data['gloss'],
            'id': idx,
            'videos_processed': processed_counts.get(sign_data['gloss'], 0)
        }
        for idx, sign_data in enumerate(wlasl_data)
    ]
    
    # Save processed sign information
    with open(os.path.join(output_path, 'processed_signs.json'), 'w') as f:
        json.dump(processed_signs, f, indent=2)
    
    print(f"Dataset processing complete. Processed {sum(results)} videos successfully.")
    return processed_signs

if __name__ == "__main__":
    process_wlasl_dataset_parallel()