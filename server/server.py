from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global vairables to track current process

@app.route('/start_translation', methods=['POST'])
def start_translation():
    """
    Endpoint to start ASL translation using provided stream url
    """
    data = request.get_json()
    stream_url = data.get('stream_url')
    print(stream_url)

    return jsonify({
        'status': 'success',
        'stream_url': stream_url,
    })
    

@app.route('/stop_translation', methods=['POST'])
def stop_translation():
    """
    Endpoint to stop ASL translation
    """
    data = request.get_json()
    stream_url = data.get('stream_url')
    print(stream_url)

@app.route('/status', methods=['GET'])
def status():
    """
    Endpoint to check if ASL Translator is running
    """
    return jsonify({
        'status': 'success',
        'running': False,
        'stream_url': None,
    })
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)