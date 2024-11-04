import os
import sys
import socket
import time
import signal

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

class PortManager:
    @staticmethod
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                return False
            except socket.error:
                return True

    @staticmethod
    def wait_for_port_available(port, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not PortManager.is_port_in_use(port):
                return True
            time.sleep(1)
        return False

    @staticmethod
    def ensure_port_available(port):
        if PortManager.is_port_in_use(port):
            print(f"Port {port} is in use, waiting for it to become available...")
            if not PortManager.wait_for_port_available(port):
                print(f"Error: Port {port} is in use and did not become available")
                return False
        return True

def setup_signal_handlers():
    def signal_handler(signum, frame):
        print("Received shutdown signal, cleaning up...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def configure_socketio():
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
        engineio_logger=True,
        always_connect=True,
        transports=['websocket', 'polling']
    )

def create_server_socket(port):
    listener = eventlet.listen(('0.0.0.0', port))
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return listener

if __name__ == "__main__":
    try:
        setup_signal_handlers()
        print("Starting 3D Avatar Application...")
        
        port = 5000
        if not PortManager.ensure_port_available(port):
            sys.exit(1)
            
        print(f"Server will be available at http://0.0.0.0:{port}")
        
        # Configure Socket.IO
        configure_socketio()
        
        # Create and configure server socket
        listener = create_server_socket(port)
        
        # Start server with improved error handling
        eventlet.wsgi.server(
            listener,
            app,
            log_output=True,
            max_size=None,
            debug=False,
            socket_timeout=60
        )
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        sys.exit(1)
