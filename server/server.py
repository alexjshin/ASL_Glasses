from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import os
import time
import subprocess
import json
import mediapipe as mp
import threading
import numpy as np
from tensorflow.keras.models import load_model
from preprocess_data import (
    get_actions,
    extract_hand_landmarks,
    extract_hand_pose_landmarks,
)
from keypoint_detection import (
    process_mp_frames,
    draw_landmarks,
    extract_keypoints_comprehensive
)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global vairables to track current process
active_translation = None
current_stream_url = None
translation_thread = None
actions = None
model = None
connected_clients = set()

def load_asl_model():
    """
    Function to load ASL model
    """
    global actions, model

    actions = get_actions('../MP_Data_01')

    print("Loading ASL model...")
    model = load_model('../Models/02_hand_pose_lstm_model.h5')
    model.load_weights('../Models/02_hand_pose_model.weights.h5')

    print("Model loaded successfully")
    return True

def get_direct_stream_url(instagram_live_url):
    """
    Use yt-dlp to get the direct streaming URL from an Instagram livestream
    """
    try:
        # Path to your cookies file - update this to your actual path
        cookies_file = os.path.join(os.getcwd(), "instagram_cookies.txt")
        
        if not os.path.exists(cookies_file):
            print(f"Cookie file not found at: {cookies_file}")
            return None
        
        # Use yt-dlp with cookies for authentication
        cmd = [
            'yt-dlp', 
            '--cookies', cookies_file,
            '--verbose',
            '-f', 'b',  # Use best available format
            '-g',       # Get direct URL only
            '--no-check-certificate',  # Skip HTTPS certificate validation if needed
            '--no-warnings',           # Reduce noise in output
            instagram_live_url
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Check for errors
        if process.returncode != 0:
            error_output = stderr.decode('utf-8')
            print(f"yt-dlp error (code {process.returncode}): {error_output}")
            return None
        
        # Get the direct URL from stdout
        direct_url = stdout.decode('utf-8').strip()
        
        if not direct_url:
            print("No direct URL found in yt-dlp output")
            return None
            
        print(f"Successfully extracted stream URL")
        return direct_url
        
    except Exception as e:
        print(f"Error getting direct stream URL: {str(e)}")
        return None

def make_prediction(sequence, threshold=0.75):
    """
    Function to make prediction using the model
    """
    global model, actions

    # Make prediction with model
    res = model.predict(np.expand_dims(sequence, axis=0))[0]
    predicted_action = actions[np.argmax(res)]
    confidence = float(res[np.argmax(res)]) # Convert numpy float to Python float

    should_use = confidence > threshold

    return {
        'predicted_action': predicted_action,
        'confidence': confidence,
        'should_use': should_use,
        'prediction_index': np.argmax(res)
    }

def asl_translation_worker(stream_url):
    """
    Worker function to handle ASL translation given a stream URL
    """
    global active_translation, current_stream_url

    direct_stream_url = get_direct_stream_url(stream_url)

    if not direct_stream_url:
        print(f"Failed to get direct stream URL for {stream_url}")
        active_translation = False
        socketio.emit('translation_error', {'message': f"Failed to access stream: {stream_url}"})
        return False

    # Intialize Mediapipe
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    # Colors for visualization (just for development)
    colors = [
        (245, 117, 16),  # Orange
        (117, 245, 16),  # Green
        (16, 117, 245),  # Blue
        (255, 0, 0),     # Red
        (0, 255, 255),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 0, 0)        # Black
    ]

    # Prediction Variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.95

    # Connect to the livestream
    try:
        cap = cv2.VideoCapture(direct_stream_url)
        if not cap.isOpened():
            print(f"Failed to open stream: {stream_url}")
            active_translation = False
            socketio.emit('translation_error', {'message': f"Failed to open stream with direct URL: {direct_stream_url}"})
            return False
    except Exception as e:
        print(f"Error opening stream: {e}")
        active_translation = False
        socketio.emit('translation_error', {'message': f"Error opening stream: {str(e)}"})
        return False
    
    print(f"Connect to stream: {stream_url}")
    socketio.emit('translation_status', {'message': f"Connected to stream: {stream_url}"})

    # Start ASL translation -- review
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while active_translation and cap.isOpened():
            # Read frame from stream
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from stream")
                socketio.emit('translation_warning', {'message': 'Failed to read frame, retrying...'})
                time.sleep(1)
                continue
            
            # Process frame with MediaPipe
            image, result = process_mp_frames(frame, holistic)

            # Draw landmarks on the frame
            draw_landmarks(image, result) 

            # Extract keypoints for prediction
            keypoints = extract_keypoints_comprehensive(result)
            keypoints = extract_hand_pose_landmarks(keypoints) # If Hand Landmarks only

            # Add to sequence and keep last 30 frames
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            # Make prediction when sequence is complete
            if len(sequence) == 30:
                prediction = make_prediction(sequence, threshold)
                predicted_action = prediction['predicted_action']
                confidence = prediction['confidence']

                print(f"Predicted Action: {predicted_action}, Confidence: {confidence}")
                predictions.append(prediction['prediction_index'])

                # Add to sentence if prediction is consistent and above threshold
                if np.unique(predictions[-10:])[0] == prediction['prediction_index']: # Check if last 10 predictions are the same
                    if prediction['should_use']:
                        if len(sentence) > 0:
                            if predicted_action != sentence[-1]:
                                sentence.append(predicted_action)
                                current_sentence = ' '.join(sentence)
                                print(f"Current sentence: {current_sentence}")
                                
                                # Send the translation to all connected clients via WebSocket
                                translation_data = {
                                    'sign': predicted_action,
                                    'confidence': confidence,
                                    'sentence': current_sentence
                                }
                                socketio.emit('translation_update', translation_data)
                        else:
                            sentence.append(predicted_action)
                            current_sentence = predicted_action
                            print(f"Current sentence: {current_sentence}")
                            
                            # Send the translation to all connected clients via WebSocket
                            translation_data = {
                                'sign': predicted_action,
                                'confidence': confidence,
                                'sentence': current_sentence
                            }
                            socketio.emit('translation_update', translation_data)
                if len(sentence) > 5:
                    sentence = sentence[-5:]
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.01)
    # Clean Up
    cap.release()
    active_translation = False
    current_stream_url = None
    socketio.emit('translation_status', {'status': 'stopped', 'message': 'ASL translation stopped'})
    print("ASL translation stopped")
    return True

def asl_translation_manager(stream_url):
    """
    Manager function that sets up and controls ASL Translation in a different thread
    """
    global active_translation, current_stream_url, translation_thread

    # If Translation is already active stop it first
    if active_translation:
        print("Stopping current ASL translation...")
        stop_translation_process()
    
    # Validate the stream URL
    if not stream_url or not isinstance(stream_url, str):
        print("Invalid stream URL provided")
        return False
    
    try:
        # Set new active stream
        active_translation = True
        current_stream_url = stream_url
        
        # Start translation in a separate thread
        translation_thread = threading.Thread(target=asl_translation_worker, args=(stream_url,))
        translation_thread.daemon = True
        translation_thread.start()
        
        # Verify the thread started successfully
        if not translation_thread.is_alive():
            print("Failed to start translation thread")
            active_translation = False
            current_stream_url = None
            return False
            
        return True
        
    except Exception as e:
        print(f"Error starting ASL translation: {e}")
        active_translation = False
        current_stream_url = None
        return False
        

def stop_translation_process():
    """
    Function to stop the current ASL translation process
    """
    global active_translation, translation_thread

    if active_translation and translation_thread:
        print("Stopping ASL translation...")
        active_translation = False

        if translation_thread.is_alive():
            translation_thread.join(timeout=5.0)
            translation_thread = None
        print("ASL translation stopped.")
        return True
    return False


###### ENDPOINTS ######

@app.route('/start_translation', methods=['POST'])
def start_translation():
    """
    Endpoint to start ASL translation using provided stream url
    """
    data = request.get_json()
    stream_url = data.get('stream_url')
    if not stream_url:
        return jsonify({
            'status': 'error',
            'message': 'Stream URL is required',
        }), 400
    else:
        print(f"Stream URL: {stream_url}")

    # Start ASL translation
    success = asl_translation_manager(stream_url)

    if success:
        return jsonify({
            'status': 'success',
            'message': 'ASL translation started successfully',
            'stream_url': stream_url,
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to start ASL translation',
        }), 500
    

@app.route('/stop_translation', methods=['POST'])
def stop_translation():
    """
    Endpoint to stop ASL translation
    """
    success = stop_translation_process()
    if success:
        return jsonify({
            'status': 'success',
            'message': 'ASL translation stopped successfully',
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to stop ASL translation',
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """
    Endpoint to check if ASL Translator is running
    """
    global active_translation, current_stream_url

    return jsonify({
        'status': 'success',
        'running': active_translation,
        'stream_url': current_stream_url,
    })

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """
    Handle client WebSocket connection
    """
    connected_clients.add(request.sid)
    print(f"Client connected: {request.sid}. Total clients: {len(connected_clients)}")
    socketio.emit('translation_status', {
        'status': 'connected',
        'running': active_translation,
        'stream_url': current_stream_url
    }, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handle client WebSocket disconnection
    """
    if request.sid in connected_clients:
        connected_clients.remove(request.sid)
    print(f"Client disconnected: {request.sid}. Total clients: {len(connected_clients)}")
    
if __name__ == "__main__":
    # Load the ASL model
    load_asl_model()

    socketio.run(app, host='0.0.0.0', port=8000, allow_unsafe_werkzeug=True)