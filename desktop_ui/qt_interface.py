import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QProgressBar, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor

class AvatarInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Avatar Control Interface')
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QLabel {
                color: #00ff00;
                font-size: 14px;
            }
            QPushButton {
                background-color: #2a2a2a;
                color: #00ff00;
                border: 2px solid #00ff00;
                border-radius: 5px;
                padding: 5px;
            }
            QProgressBar {
                border: 2px solid #00ff00;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #00ff00;
            }
        """)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add widgets
        self.response_label = QLabel("Waiting for command...")
        self.start_button = QPushButton("Start Interaction")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)

        layout.addWidget(self.response_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.progress_bar)

        # Connect signals
        self.start_button.clicked.connect(self.start_interaction)

        self.setMinimumSize(400, 200)

    def start_interaction(self):
        self.progress_bar.setValue(0)
        QTimer.singleShot(100, self.update_progress)

    def update_progress(self):
        current = self.progress_bar.value()
        if current < 100:
            self.progress_bar.setValue(current + 1)
            QTimer.singleShot(50, self.update_progress)

def launch_qt_interface():
    app = QApplication(sys.argv)
    window = AvatarInterface()
    window.show()
    sys.exit(app.exec_())
