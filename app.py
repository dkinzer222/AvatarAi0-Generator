import os
from flask import Flask, render_template
from flask_socketio import SocketIO
import json

app = Flask(__name__)
socketio = SocketIO(app)

# Load credentials from env
with open('env.txt') as f:
    env_data = json.load(f)
    
app.config['SECRET_KEY'] = env_data['FLASK_SECRET_KEY']
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json.dumps(env_data['SERVICE_ACCOUNT_JSON'])

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('pose_data')
def handle_pose_data(data):
    # Process pose data and update avatar
    socketio.emit('avatar_update', data)

@socketio.on('voice_command')
def handle_voice(data):
    from utils.google_cloud import process_speech
    response = process_speech(data)
    socketio.emit('voice_response', response)
