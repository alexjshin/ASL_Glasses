import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

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
                    res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                    window.append(res)
                except Exception as e:
                    print(f"Error loading file: {os.path.join(DATA_PATH, action, str(sequence), f'{frame_num}.npy')}")
                    print(f"Error details: {e}")
                    # If a frame is missing, you can choose to skip or use zeros
                    # Here we'll use zeros to maintain sequence length
                    window.append(np.zeros(res.shape if 'res' in locals() else 1662))
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