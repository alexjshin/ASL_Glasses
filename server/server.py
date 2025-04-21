from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import tempfile
import os
import time
import subprocess
import mediapipe as mp
import threading
import numpy as np
from obswebsocket import obsws, requests
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
active_translation = False
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

def launch_obs():
    try:
        subprocess.Popen(["open", "-a", "OBS"])
        print("✅ OBS launched")
        return True
    except Exception as e:
        print(f"❌ OBS launch failed: {e}")
        return False

def start_obs_stream_via_websocket(host="localhost", port=4455, password="cpsc490"):
    try:
        time.sleep(10)  # Give OBS time to fully launch
        ws = obsws(host, port, password)
        ws.connect()
        print("✅ Connected to OBS WebSocket")

        try:
            ws.call(requests.StartStreaming())
            print("🎥 Streaming started")
        except Exception as e:
            print(f"Note: {e} - OBS may already be streaming")
            
        ws.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ WebSocket start failed: {e}")
        return False

def translation_worker():
    global active_translation
    sequence, sentence, predictions = [], [], []
    
    # Define ROI coordinates
    roi_x = 0      
    roi_y = 38      
    roi_width = 498  
    roi_height = 1117 
    
    print(f"Using ROI: x={roi_x}, y={roi_y}, width={roi_width}, height={roi_height}")
    socketio.emit("translation_status", {"status": "running", "message": "Starting screen capture with defined ROI."})
    
    def capture_screen_roi():
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()
        
        # Capture screen to the temporary file
        subprocess.call(['screencapture', '-x', temp_file.name])
        
        # Read the image
        full_frame = cv2.imread(temp_file.name)
        
        # Crop to ROI
        if full_frame is not None:
            # Make sure ROI is within the image bounds
            max_y, max_x = full_frame.shape[:2]
            if roi_x < max_x and roi_y < max_y:
                # Ensure we don't go out of bounds
                end_x = min(roi_x + roi_width, max_x)
                end_y = min(roi_y + roi_height, max_y)
                roi_frame = full_frame[roi_y:end_y, roi_x:end_x]
            else:
                roi_frame = full_frame  # Fallback to full frame if coordinates are invalid
        else:
            roi_frame = None
        
        # Clean up
        os.unlink(temp_file.name)
        
        return roi_frame
    
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while active_translation:
            # Capture screen ROI
            frame = capture_screen_roi()
            
            if frame is None or frame.size == 0:
                print("⚠️ Failed to capture screen ROI. Retrying...")
                time.sleep(1)
                continue
            
            # Process frame with MediaPipe
            image, results = process_mp_frames(frame, holistic)
            draw_landmarks(image, results)
            keypoints = extract_hand_pose_landmarks(extract_keypoints_comprehensive(results))

            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                prediction = make_prediction(sequence, threshold=0.95)
                predictions.append(prediction['prediction_index'])

                if (np.unique(predictions[-10:])[0] == prediction['prediction_index']
                    and prediction['should_use']):
                    if not sentence or prediction['predicted_action'] != sentence[-1]:
                        sentence.append(prediction['predicted_action'])
                        sentence = sentence[-5:]
                        socketio.emit("translation_update", {
                            "sign": prediction['predicted_action'],
                            "confidence": prediction['confidence'],
                            "sentence": " ".join(sentence)
                        })
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.1)

    active_translation = False
    socketio.emit("translation_status", {"status": "stopped", "message": "Translation finished."})

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
    global active_translation, translation_thread

    if active_translation:
        stop_translation_process()

    active_translation = True

    # if not launch_obs():
    #     return jsonify({"status": "error", "message": "Failed to launch OBS"}), 500
    # if not start_obs_stream_via_websocket():
    #     return jsonify({"status": "error", "message": "Failed to start OBS stream"}), 500

    translation_thread = threading.Thread(target=translation_worker)
    translation_thread.start()

    return jsonify({"status": "success", "message": "ASL translation started."}), 200

    

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