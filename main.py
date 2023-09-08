import sys
import cv2
import time
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDockWidget, QPushButton, QLineEdit)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from detector import Detector

class EyeTrackingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.lock_status = False
        self.lock_point = None
        self.lock_diameter = None
        self.lock_start_time = None
        self.deviation_record = None
        self.time_record = None

        self.pause_status = False

        self.detector = Detector()
        self.camera = cv2.VideoCapture(0)

        # Create video capture widget
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(960, 640)

        # Create plot widget
        self.plot_widget = QWidget(self)
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_canvas = FigureCanvas(plt.Figure())
        self.plot_layout.addWidget(self.plot_canvas)

        # Create buttons
        self.LockBTN = QPushButton("Lock")
        self.LockBTN.clicked.connect(lambda: self.lock(100))
        self.PauseBTN = QPushButton("Pause")
        self.PauseBTN.clicked.connect(self.pause)
        self.UnpauseBTN = QPushButton("Unpause")
        self.UnpauseBTN.clicked.connect(self.unpause)
        self.UnlockBTN = QPushButton("Unlock")
        self.UnlockBTN.clicked.connect(self.unlock)
        self.SaveBTN = QPushButton("Save")
        self.SaveBTN.clicked.connect(self.save_results)

        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(100,40)

        # Create button widget
        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.addWidget(self.LockBTN)
        self.button_layout.addWidget(self.PauseBTN)
        self.button_layout.addWidget(self.UnpauseBTN)
        self.button_layout.addWidget(self.UnlockBTN)
        self.button_layout.addWidget(self.SaveBTN)
        self.button_layout.addWidget(self.textbox)

        # Set layout and add widget into the layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.button_widget)

        dockWidget = QDockWidget("Deviation", self)
        dockWidget.setAllowedAreas(Qt.TopDockWidgetArea | Qt.RightDockWidgetArea)
        dockWidget.setWidget(self.plot_canvas)
        self.addDockWidget(Qt.RightDockWidgetArea, dockWidget)

        main_widget = QWidget(self)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Self-refresg timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

        self.setWindowTitle("Eye Tracking")
        self.setGeometry(100, 100, 1000, 500)

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret and not self.pause_status:
            Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FlippedImage = cv2.flip(Image, 1)

            self.detector.detect(FlippedImage)
            FlippedImage = self.detector.label(FlippedImage, self.lock_status, self.lock_point, self.lock_diameter)


            ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(960, 640, Qt.KeepAspectRatio)
            pixmap = QPixmap.fromImage(Pic)
            self.video_label.setPixmap(pixmap)

            self.current_point = self.detector.pupil_left if self.detector.face_detected else None # Left pupil point
            self.current_diameter = self.detector.pupil_diameter


            if self.lock_status:
                t = round(time.time() - self.lock_start_time,2)

                self.time_record = np.append(self.time_record, t)
                self.deviation_record = np.append(self.deviation_record, self.detector.deviation_left)

                self.plot_deviation(self.time_record, self.deviation_record)

    def plot_deviation(self, time_record, deviation_record):
        self.plot_canvas.figure.clear()
        ax = self.plot_canvas.figure.add_subplot(111)
        ax.plot(time_record, deviation_record)
        # ax.hist(g_values, bins=20, color='green', alpha=0.5, label='Green')
        # ax.hist(b_values, bins=20, color='blue', alpha=0.5, label='Blue')
        ax.set_title('Target Deviation')
        ax.set_xlabel('Time')
        ax.set_ylabel('Distance')
        self.plot_canvas.draw()

    def lock(self, size=100):
        if self.detector.face_detected:
            self.lock_start_time = time.time()
            self.lock_status = True
            self.lock_point = self.current_point # Left pupil point
            self.lock_diameter = int(self.current_diameter * size/100) #The diameter for the central pupil locking area as the percentage of the iris diameter
            self.deviation_record = np.array(0)
            self.time_record = np.array(0)

    def unlock(self):
        self.lock_status = False

    def pause(self):
        self.pause_status = True

    def unpause(self):
        self.pause_status = False

    def save_results(self):
        df = pd.DataFrame(
            {
                't':self.time_record,
                'deviation':self.deviation_record
            }
        )
        file_name = self.textbox.text()
        try:
            df.to_csv(f'{file_name}.csv', index=False)
            self.textbox.setText("")
        except:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeTrackingApp()
    window.show()
    sys.exit(app.exec_())
