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

# Configure eventlet before any other imports
import eventlet
eventlet.monkey_patch()

from app import app, socketio

if __name__ == "__main__":
    try:
        print("Starting 3D Avatar Application...")
        print("Server will be available at http://0.0.0.0:5000")
        
        # Use eventlet wsgi server
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            log_output=True
        )
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        raise
