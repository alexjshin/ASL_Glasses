import cv2
import numpy as np
import mediapipe as mp
from keypoint_detection import (
    process_mp_frames,
    draw_landmarks,
    extract_keypoints_comprehensive)
from preprocess_data import preprocess_data, extract_hand_landmarks, extract_hand_pose_landmarks
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

print("testing model")

######### PREPROCESS DATA AND CREATE LABELS AND FEATURES #########
X_train, X_test, y_train, y_test, actions = preprocess_data('MP_Data_01', sequence_length=30)

######### LOAD MODEL AND TEST REALTIME #########
model = load_model('../Models/02_hand_pose_lstm_model.h5')
model.load_weights('../Models/02_hand_pose_model.weights.h5')
print("Model loaded successfully")

colors = [
    (245, 117, 16),  # Orange
    (117, 245, 16),  # Green
    (16, 117, 245),  # Blue
    (255, 0, 0),     # Red
    (0, 255, 255),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 0, 0)        # Black
]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = process_mp_frames(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints_comprehensive(results)
        keypoints = extract_hand_pose_landmarks(keypoints) # If Hand Landmarks only
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


