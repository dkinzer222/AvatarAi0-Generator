from app import app
from desktop_ui.qt_interface import launch_qt_interface
import threading

if __name__ == "__main__":
    # Start Qt interface in a separate thread
    qt_thread = threading.Thread(target=launch_qt_interface)
    qt_thread.daemon = True
    qt_thread.start()
    
    # Start Flask server
    app.run(host="0.0.0.0", port=5000, debug=True)
