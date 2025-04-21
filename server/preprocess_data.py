import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def get_actions(data_path):
    """
    Get all action classes from the dataset directory.
    
    Args:
        data_path: Path to the dataset directory
        
    Returns:
        actions: List of action classes
    """
    DATA_PATH = os.path.join(data_path)
    # Automatically detect action classes from directory structure
    actions = np.array([folder for folder in os.listdir(DATA_PATH) if not folder.startswith('.DS_Store')]) # .DS_Store is issue
    # print(f"Detected actions: {actions}")
    return actions

def extract_hand_landmarks(landmarks_data, hand_only=True):
    """
    Extract only hand landmarks from the full MediaPipe data.
    
    MediaPipe typically provides landmarks in this order:
    - Pose: 33 landmarks x 4 values (x, y, z, visibility) = 132 values
    - Face: 468 landmarks x 3 values (x, y, z) = 1404 values
    - Left hand: 21 landmarks x 3 values (x, y, z) = 63 values
    - Right hand: 21 landmarks x 3 values (x, y, z) = 63 values
    
    Total: 1662 values (which matches your default dimension)
    
    Args:
        landmarks_data: Full landmark data from MediaPipe
        hand_only: If True, extract only hand landmarks
        
    Returns:
        Hand landmarks data
    """
    if hand_only:
        # Using the expected indices based on MediaPipe's structure
        # Pose (132) + Face (1404) = 1536, then hands start
        left_hand_start = 1536  # After pose and face
        right_hand_start = left_hand_start + 63  # After left hand
        
        # Extract both hands (63 values each, total 126)
        left_hand = landmarks_data[left_hand_start:right_hand_start]
        right_hand = landmarks_data[right_hand_start:right_hand_start+63]
        
        # Combine both hands
        return np.concatenate([left_hand, right_hand])
    else:
        # Return full data
        return landmarks_data

def extract_hand_pose_landmarks(landmarks_data, hand_pose_only=True):
    """
    Extract only hand and pose landmarks from the full MediaPipe data.
    
    MediaPipe typically provides landmarks in this order:
    - Pose: 33 landmarks x 4 values (x, y, z, visibility) = 132 values
    - Face: 468 landmarks x 3 values (x, y, z) = 1404 values
    - Left hand: 21 landmarks x 3 values (x, y, z) = 63 values
    - Right hand: 21 landmarks x 3 values (x, y, z) = 63 values
    
    Total: 1662 values (which matches your default dimension)
    
    Args:
        landmarks_data: Full landmark data from MediaPipe
        hand_only: If True, extract only hand landmarks
        
    Returns:
        Hand landmarks data
    """
    if hand_pose_only:
        # Using the expected indices based on MediaPipe's structure
        # Pose (132) + Face (1404) = 1536, then hands start
                # Extract pose landmarks (first 132 values)
        # Extract pose landmarks (first 132 values)
        pose_landmarks = landmarks_data[:132]
        left_hand_start = 1536  # After pose and face
        right_hand_start = left_hand_start + 63  # After left hand
        
        # Extract both hands (63 values each, total 126)
        left_hand = landmarks_data[left_hand_start:right_hand_start]
        right_hand = landmarks_data[right_hand_start:right_hand_start+63]
        
        # Combine both hands
        return np.concatenate([pose_landmarks, left_hand, right_hand])
    else:
        # Return full data
        return landmarks_data

def preprocess_data(data_path, sequence_length=30):
    DATA_PATH = os.path.join(data_path)
    # Automatically detect action classes from directory structure
    actions = np.array([folder for folder in os.listdir(DATA_PATH) if not folder.startswith('.DS_Store')]) # .DS_Store is issue
    print(f"Detected actions: {actions}")

    # Create label mapping
    label_map = {label: num for num, label in enumerate(actions)}
    print(f"Label mapping: {label_map}")

    # Initialize empty lists for sequences and labels
    sequences, labels = [], []

    # Loop through each action
    for action in actions:
        # Loop through all sequences
        sequence_folders = [folder for folder in os.listdir(os.path.join(DATA_PATH, action)) if not folder.startswith('.DS_Store')]
        for sequence in np.array(sequence_folders).astype(int):
            # Build window of frames for this sequence
            window = []
            for frame_num in range(sequence_length):
                # Load the frame data (keypoints)
                try:
                    full_landmarks = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                    # hand_landmarks = extract_hand_landmarks(full_landmarks)
                    hand_pose_landmarks = extract_hand_pose_landmarks(full_landmarks)
                    window.append(hand_pose_landmarks)
                    # res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                    # window.append(res)
                except Exception as e:
                    print(f"Error loading file: {os.path.join(DATA_PATH, action, str(sequence), f'{frame_num}.npy')}")
                    print(f"Error details: {e}")
                    # If a frame is missing, you can choose to skip or use zeros
                    # Here we'll use zeros to maintain sequence length
                    # for full landmarks
                    # window.append(np.zeros(res.shape if 'res' in locals() else 1662))

                    # for hand landmarks
                    window.append(np.zeros(126))  # 21 landmarks * 3 values (x, y, z) for both hands
                    
                    # for hand_pose landmarks
                    window.append(np.zeros(126 + 132))  # 21 landmarks * 3 values (x, y, z) for both hands + pose landmarks

            # Add this sequence and its label to our datasets - FOR WLASL -> may not have complete sequence
            if len(window) == sequence_length:  # Ensure we have a complete sequence
                sequences.append(window)
                labels.append(label_map[action])
    # Convert sequences and labels to numpy arrays
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    print(f"Data shape: {X.shape}")  # Should be (num_sequences, sequence_length, num_features)
    print(f"Labels shape: {y.shape}")  # Should be (num_sequences, num_classes)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, actions