import os
# Force CPU/Software rendering for better compatibility
os.environ['OPENCV_VIDEOIO_MMAP'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['PYTHONPATH'] = '.'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_DEBUG'] = '0'

# Configure eventlet before any other imports
import eventlet
eventlet.monkey_patch()

from app import app, socketio

if __name__ == "__main__":
    try:
        print("Starting 3D Avatar Application...")
        print("Server will be available at http://0.0.0.0:5000")
        
        # Enhanced Socket.IO configuration
        socketio.init_app(
            app,
            cors_allowed_origins="*",
            ping_timeout=60,
            ping_interval=25,
            async_mode='eventlet',
            reconnection=True,
            reconnection_attempts=5,
            reconnection_delay=1000,
            reconnection_delay_max=5000,
            logger=True,
            engineio_logger=True
        )
        
        # Use eventlet wsgi server with improved configuration
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            log_output=True,
            allow_unsafe_werkzeug=True
        )
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        raise
