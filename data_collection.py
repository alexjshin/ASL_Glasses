import cv2
import os
import logging
import time
import numpy as np
import mediapipe as mp
import json
from tqdm import tqdm # for progress bars in Jupyter Notebook
from matplotlib import pyplot as plt
from keypoint_detection import (
    setup_holistic_detection,
    process_mp_frames,
    draw_landmarks,
    extract_keypoints_comprehensive,
    create_directories)

# Create Directories to store data
def create_directories(actions):
    DATA_PATH = os.path.join('MP_Data')

    # Detecting Actions
    for action in actions:
        # 30 is often considered minimum threshold to detect meaningful patterns
        # n >= 30 commonly used as rule of thumb
        for sequence in range(30): # 30 videos per action 
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def collect_asl_data():
    # Define Actions - starting with a small set for testing
    actions = np.array(['hello', 'thanks', ' I love you', 'yes', 'no', 'please', 'sorry'])

    # create directories for data collection - 30 directories per action
    create_directories(actions)

    # Define data path
    DATA_PATH = os.path.join('MP_Data')

    # set up holistic model
    cap, holistic = setup_holistic_detection()

    # Loop through different actions
    for action in actions:
        # loop through sequences
        for sequence in range(30):
            # loop through video length of 30 frames
            for frame_num in range(30):
                # Read Feed
                ret, frame = cap.read()
                if not ret:
                    print('Failed to grab frame')
                    break

                # Process the frame for landmark detection
                image, results = process_mp_frames(frame, holistic)

                # Draw Landmarks on the image
                draw_landmarks(image, results)

                # Apply status text to the frame
                if frame_num == 0:
                    # Visual Countdown before starting each sequence
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting for {action} - Sequence {sequence}', (15, 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('ASL Data Collection', image)
                    cv2.waitKey(2000)  # Wait 2 seconds before starting
                else:
                    # Display collection status
                    cv2.putText(image, f'Collecting for {action} - Sequence {sequence}', (15, 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Frame {frame_num}', (15, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # show to screen
                cv2.imshow('ASL Data Collection', image)

                # Export keypoints
                keypoints_vector = extract_keypoints_comprehensive(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints_vector)
                
                # Break gracefully on 'q' press
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            # Short break between sequences
            if sequence < 29:  # Don't show after the last sequence
                # Display break message
                ret, frame = cap.read()
                if ret:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.putText(image, 'SEQUENCE COMPLETE', (120, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Preparing for next sequence...', (120, 230), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow('ASL Data Collection', image)
                cv2.waitKey(2000)  # 2 second break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete!")