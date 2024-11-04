import os
import sys
import socket
import time
import signal
import atexit
import logging
import eventlet
import threading
from concurrent.futures import ThreadPoolExecutor

# Force CPU/Software rendering for better compatibility
os.environ['OPENCV_VIDEOIO_MMAP'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_DEBUG'] = '0'

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Configure eventlet with timeout
eventlet.monkey_patch(socket=True, select=True)

from flask import Flask
from flask_socketio import SocketIO

def verify_dependencies():
    """Verify all required dependencies are available"""
    try:
        import cv2
        import numpy
        import mediapipe
        logger.info("All core dependencies verified")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        return False

def create_app():
    """Initialize Flask app with enhanced error handling"""
    try:
        # Initialize Flask app
        app = Flask(__name__)
        app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        
        # Configure Socket.IO with improved settings
        socketio = SocketIO(
            app,
            async_mode='eventlet',
            logger=True,
            engineio_logger=True,
            cors_allowed_origins="*",
            ping_timeout=60,
            ping_interval=25,
            max_http_buffer_size=10e6,
            manage_session=False,
            always_connect=True
        )
        
        # Import and configure routes
        from app import configure_routes
        configure_routes(app, socketio)
        
        logger.info("Application created successfully")
        return app, socketio
        
    except Exception as e:
        logger.error(f"Error creating application: {str(e)}")
        raise

def cleanup_port(port, max_retries=3):
    """Force cleanup of the port with retry mechanism"""
    for attempt in range(max_retries):
        try:
            # Create a new socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(1)
            
            # Try to bind to the port
            sock.bind(('0.0.0.0', port))
            sock.close()
            
            # Wait for socket to fully close
            time.sleep(1)
            logger.info(f"Successfully cleaned up port {port}")
            return True
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} to clean port {port} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            else:
                logger.error(f"Failed to clean up port {port} after {max_retries} attempts")
                return False
        finally:
            try:
                sock.close()
            except:
                pass

def setup_signal_handlers(socketio_instance):
    """Set up signal handlers for graceful shutdown"""
    def cleanup():
        logger.info("Cleaning up resources...")
        try:
            socketio_instance.stop()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        cleanup()
        sys.exit(0)

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def initialize_server():
    """Initialize server components"""
    try:
        # Verify dependencies
        if not verify_dependencies():
            raise RuntimeError("Required dependencies not available")
        
        # Create Flask app and Socket.IO instance
        app, socketio = create_app()
        
        # Set up signal handlers
        setup_signal_handlers(socketio)
        
        return app, socketio
    except Exception as e:
        logger.error(f"Server initialization failed: {str(e)}")
        raise

def start_server(host='0.0.0.0', port=5000):
    """Start the server with enhanced error handling and monitoring"""
    try:
        # Initialize server components
        app, socketio = initialize_server()
        
        # Clean up the port
        if not cleanup_port(port):
            raise RuntimeError(f"Failed to clean up port {port}")
        
        # Create a monitoring thread
        def monitor_server():
            while True:
                try:
                    # Check if port is still bound
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    if result != 0:
                        logger.error("Server port check failed")
                        break
                        
                    time.sleep(5)  # Check every 5 seconds
                except:
                    break

        monitor_thread = threading.Thread(target=monitor_server, daemon=True)
        monitor_thread.start()
        
        # Start server
        logger.info(f"Starting server on {host}:{port}")
        socketio.run(
            app,
            host=host,
            port=port,
            debug=False,
            use_reloader=False,
            log_output=True
        )
        return True
        
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        logger.info("Starting 3D Avatar Application...")
        port = int(os.environ.get('PORT', 5000))
        
        # Start the server
        if not start_server(port=port):
            logger.error("Failed to start server")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
