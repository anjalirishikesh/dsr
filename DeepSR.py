import os
import sys
import time
import shutil
import math
import ffmpeg
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip
import subprocess
from typing import Tuple
import vidViewer
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QDir, Qt, QUrl, QSize
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, 
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QStatusBar)

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QAction

import progressbar
import easydict
import utils
from PIL import Image

import random
import re
import os
import tempfile
import ssl
import cv2
import scipy.special
import numpy as np
import pickle as pkl
import math
import copy
import tensorflow as tf
import os
from urllib import request  # requires python3


interpreter = tf.lite.Interpreter(model_path="onnx_lite64.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_label():
  with open("labels.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    return labels

# load the label
word_data = load_label(
    
  )
# load the tensorflow lite run time

dir = os.getcwd()
class model_loader(QtCore.QThread):
    report = QtCore.pyqtSignal(int)
    def __init__(self,inputs,outputs):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
     

    def run(self):
        args = easydict.EasyDict({"input":self.inputs,"output":self.outputs})
        is_video = False
        if not os.path.exists(args.input):
            print('Error: Folder [{:s}] does not exist.'.format(args.input))
            sys.exit(1)
        elif os.path.isfile(args.input) and args.input.split('.')[-1].lower() in ['mp4', 'mkv', 'm4v', 'gif']:
            is_video = True
            if args.output.split('.')[-1].lower() not in ['mp4', 'mkv', 'm4v', 'gif']:
                print('Error: Output [{:s}] is not a file.'.format(args.input))
                sys.exit(1)
        elif not os.path.isfile(args.input) and not os.path.isfile(args.output) and not os.path.exists(args.output):
            os.mkdir(args.output)
     
        # resize
        resize = (224,224)
        path_input = args.input
        # read the video
        cap = cv2.VideoCapture(path_input)
        
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("the total frames",length)
        val1 = int(length/64)
        print("the ratio is :",val1)

        # Get the Default resolutions
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # Define the codec and filename.
        out = cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))
        frames = []

        in_frames = 65
        counter = 1
        frame_counter = 1
        try:
            while True:
                ret, frame_old = cap.read()
                if not ret:
                    break
                frame_new = copy.deepcopy(frame_old)
                frame = crop_center_square(frame_old)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]] 
                frames.append(frame)
                # check the counter is not zero and append the frames
                if counter!=0:
                    if counter % in_frames == 0:
                        
                        del frames[:]
                else:
                    frames.append(frame)
                res = np.array(frames) / 255.0
                print(res.shape)
                value = res.shape[0] 
                counter+=1
                # Remove the frame with zero clip
                if value !=0 and value == 64:
                    perc = frame_counter / val1
                    if frame_counter < val1:
                        perc = perc*100
                        frame_counter +=1
                        self.report.emit(perc)
                   
                    #print(res.shape)
                    model_input = tf.constant(res, dtype=tf.float32)[tf.newaxis, ...]
                    print(model_input.shape)
                    inp = tf.transpose(model_input, perm=[0, 4, 1, 2,3])
                    print(inp.shape)
                    interpreter.set_tensor(input_details[0]['index'], inp)
                    interpreter.invoke()
                    outputs = interpreter.get_tensor(output_details[0]['index'])
                    #outputs = model(inp)
                    topk=1
                    result = []
                    num_clips =1
                    num_detections = 2000
                    for i in range(num_detections):

                        res = outputs[0][i]
                        result.append(res)
                    #print(result)
                    res = []
                    res.append(result)
                    
                    raw_scores = np.array(res)

                    prob_scores = scipy.special.softmax(raw_scores, axis=1)
                    prob_sorted = np.sort(prob_scores, axis=1)[:, ::-1]
                    pred_sorted = np.argsort(prob_scores, axis=1)[:, ::-1]
                    



                    word_topk = None
                    for k in range(topk):

                        for i, p in enumerate(pred_sorted[:, k]):
                            if str(word_data[p])!="want":
                                print(str(word_data[p]))
                            
                                word_topk= word_data[p]
                    prob_topk = prob_sorted[:, :topk].transpose()
                    print("Predicted signs:")
                    print(word_topk)
                    print(type(prob_topk))


                    cv2.putText(frame_new, str(word_topk),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                if frame_counter == val1:
                    cap.release()
                    perc = 100
                    self.report.emit(perc)

                    
                    
                    

                # All the results have been drawn on the frame, so it's time to display it.
                out.write(frame_new)
        finally:
         
            cap.release()
        
            
            
class VideoPlayer(QWidget):
    

    def __init__(self, parent=None, view: vidViewer.ImageViewer=None):
        super(VideoPlayer, self).__init__(parent,QtCore.Qt.Window)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)



        btnSize = QSize(16, 16)
        videoWidget = QVideoWidget()

        openButton = QPushButton("Play")   
        openButton.setToolTip("play")
        openButton.setStatusTip("Open Video File")
        openButton.setFixedHeight(24)
        openButton.setIconSize(btnSize)
        openButton.setFont(QFont("Noto Sans", 8))
        openButton.setShortcut('Ctrl+O')
        openButton.setIcon(QtGui.QIcon('./icons/play-arrow.png'))
        openButton.clicked.connect(self.abrir)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setFixedHeight(24)
        self.playButton.setIconSize(btnSize)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Noto Sans", 7))
        self.statusBar.setFixedHeight(14)

        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(openButton)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.statusBar)

        self.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        self.statusBar.showMessage("Ready")

    def abrir(self):
        dir = os.getcwd()
        file = open("file.txt","r")
        fileName = file.read()
        #fileName = os.path.join(dir + "/" + file)
     
        if fileName != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.statusBar.showMessage(fileName)
            self.play()

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())

class MainWindow(QtWidgets.QMainWindow):
    _file_path: str = None
    videoplayer: VideoPlayer = None

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Deep Sign Recognition')
        self.setStyleSheet("QToolTip\n"
"{\n"
"     border: 1px solid black;\n"
"     background-color: #ffa02f;\n"
"     padding: 1px;\n"
"     border-radius: 3px;\n"
"     opacity: 100;\n"
"}\n"
"\n"
"QWidget\n"
"{\n"
"    color: #b1b1b1;\n"
"    background-color: #323232;\n"
"    font-size: 14px;\n"
"\n"
"}\n"
"\n"
"QWidget:item:hover\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #ca0619);\n"
"    color: #000000;\n"
"}\n"
"\n"
"QWidget:item:selected\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);\n"
"}\n"
"\n"
"QMenuBar::item\n"
"{\n"
"    background: transparent;\n"
"}\n"
"\n"
"QMenuBar::item:selected\n"
"{\n"
"    background: transparent;\n"
"    border: 1px solid #ffaa00;\n"
"}\n"
"\n"
"QMenuBar::item:pressed\n"
"{\n"
"    background: #444;\n"
"    border: 1px solid #000;\n"
"    background-color: QLinearGradient(\n"
"        x1:0, y1:0,\n"
"        x2:0, y2:1,\n"
"        stop:1 #212121,\n"
"        stop:0.4 #343434/*,\n"
"        stop:0.2 #343434,\n"
"        stop:0.1 #ffaa00*/\n"
"    );\n"
"    margin-bottom:-1px;\n"
"    padding-bottom:1px;\n"
"}\n"
"\n"
"QMenu\n"
"{\n"
"    border: 1px solid #000;\n"
"}\n"
"\n"
"QMenu::item\n"
"{\n"
"    padding: 2px 20px 2px 20px;\n"
"}\n"
"\n"
"QMenu::item:selected\n"
"{\n"
"    color: #000000;\n"
"}\n"
"\n"
"QWidget:disabled\n"
"{\n"
"    color: #404040;\n"
"    background-color: #323232;\n"
"}\n"
"\n"
"QAbstractItemView\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #4d4d4d, stop: 0.1 #646464, stop: 1 #5d5d5d);\n"
"}\n"
"\n"
"QWidget:focus\n"
"{\n"
"    /*border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);*/\n"
"}\n"
"\n"
"QLineEdit\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #4d4d4d, stop: 0 #646464, stop: 1 #5d5d5d);\n"
"    padding: 1px;\n"
"    border-style: solid;\n"
"    border: 1px solid #1e1e1e;\n"
"    border-radius: 5;\n"
"}\n"
"\n"
"QPushButton\n"
"{\n"
"    color: #b1b1b1;\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #565656, stop: 0.1 #525252, stop: 0.5 #4e4e4e, stop: 0.9 #4a4a4a, stop: 1 #464646);\n"
"    border-width: 1px;\n"
"    border-color: #1e1e1e;\n"
"    border-style: solid;\n"
"    border-radius: 10;\n"
"    padding: 3px;\n"
"    font-size: 18px;\n"
"    padding-left: 5px;\n"
"    padding-right: 8px;\n"
"}\n"
"\n"
"QPushButton:pressed\n"
"{\n"
"    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2d2d2d, stop: 0.1 #2b2b2b, stop: 0.5 #292929, stop: 0.9 #282828, stop: 1 #252525);\n"
"}\n"
"\n"
"QComboBox\n"
"{\n"
"    selection-background-color: #3D7848;\n"
"    background-color: #3D7848;\n"
"    border-style: solid;\n"
"    border: 1px solid #1e1e1e;\n"
"    border-radius: 5;\n"
"    font-size: 18px;\n"
"\n"
"}\n"
"\n"
"QComboBox:hover,QPushButton:hover\n"
"{\n"
"    border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);\n"
"}\n"
"\n"
"\n"
"QComboBox:on\n"
"{\n"
"    padding-top: 3px;\n"
"    padding-left: 4px;\n"
"    background-color: #3D7848 ;\n"
"    selection-background-color: #3D7848;\n"
"}\n"
"\n"
"QComboBox QAbstractItemView\n"
"{\n"
"    border: 2px solid darkgray;\n"
"    selection-background-color: #3D7848;\n"
"}\n"
"\n"
"QComboBox::drop-down\n"
"{\n"
"     subcontrol-origin: padding;\n"
"     subcontrol-position: top right;\n"
"     width: 15px;\n"
"\n"
"     border-left-width: 0px;\n"
"     border-left-color: darkgray;\n"
"     border-left-style: solid; /* just a single line */\n"
"     border-top-right-radius: 3px; /* same radius as the QComboBox */\n"
"     border-bottom-right-radius: 3px;\n"
" }\n"
"\n"
"QComboBox::down-arrow\n"
"{\n"

"}\n"
"\n"
"QGroupBox:focus\n"
"{\n"
"border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);\n"
"}\n"
"\n"
"QTextEdit:focus\n"
"{\n"
"    border: 2px solid QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);\n"
"}\n"
"\n"
"QScrollBar:horizontal {\n"
"     border: 1px solid #222222;\n"
"     background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0.0 #121212, stop: 0.2 #282828, stop: 1 #484848);\n"
"     height: 7px;\n"
"     margin: 0px 16px 0 16px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal\n"
"{\n"
"      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #ffa02f, stop: 0.5 #d7801a, stop: 1 #ffa02f);\n"
"      min-height: 20px;\n"
"      border-radius: 2px;\n"
"}\n"
"\n"
"QScrollBar::add-line:horizontal {\n"
"      border: 1px solid #1b1b19;\n"
"      border-radius: 2px;\n"
"      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #ffa02f, stop: 1 #d7801a);\n"
"      width: 14px;\n"
"      subcontrol-position: right;\n"
"      subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal {\n"
"      border: 1px solid #1b1b19;\n"
"      border-radius: 2px;\n"
"      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #ffa02f, stop: 1 #d7801a);\n"
"      width: 14px;\n"
"     subcontrol-position: left;\n"
"     subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::right-arrow:horizontal, QScrollBar::left-arrow:horizontal\n"
"{\n"
"      border: 1px solid black;\n"
"      width: 1px;\n"
"      height: 1px;\n"
"      background: white;\n"
"}\n"
"\n"
"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal\n"
"{\n"
"      background: none;\n"
"}\n"
"\n"
"QScrollBar:vertical\n"
"{\n"
"      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0.0 #121212, stop: 0.2 #282828, stop: 1 #484848);\n"
"      width: 7px;\n"
"      margin: 16px 0 16px 0;\n"
"      border: 1px solid #222222;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical\n"
"{\n"
"      background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 0.5 #d7801a, stop: 1 #ffa02f);\n"
"      min-height: 20px;\n"
"      border-radius: 2px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical\n"
"{\n"
"      border: 1px solid #1b1b19;\n"
"      border-radius: 2px;\n"
"      background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ffa02f, stop: 1 #d7801a);\n"
"      height: 14px;\n"
"      subcontrol-position: bottom;\n"
"      subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical\n"
"{\n"
"      border: 1px solid #1b1b19;\n"
"      border-radius: 2px;\n"
"      background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #d7801a, stop: 1 #ffa02f);\n"
"      height: 14px;\n"
"      subcontrol-position: top;\n"
"      subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical\n"
"{\n"
"      border: 1px solid black;\n"
"      width: 1px;\n"
"      height: 1px;\n"
"      background: white;\n"
"}\n"
"\n"
"\n"
"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical\n"
"{\n"
"      background: none;\n"
"}\n"
"\n"
"QTextEdit\n"
"{\n"
"    background-color: #242424;\n"
"}\n"
"\n"
"QPlainTextEdit\n"
"{\n"
"    background-color: #242424;\n"
"}\n"
"\n"
"QHeaderView::section\n"
"{\n"
"    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #616161, stop: 0.5 #505050, stop: 0.6 #434343, stop:1 #656565);\n"
"    color: white;\n"
"    padding-left: 4px;\n"
"    border: 1px solid #6c6c6c;\n"
"}\n"
"\n"
"QCheckBox:disabled\n"
"{\n"
"color: #3D7848;\n"
"}\n"
"\n"
"QDockWidget::title\n"
"{\n"
"    text-align: center;\n"
"    spacing: 3px; /* spacing between items in the tool bar */\n"
"    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #323232, stop: 0.5 #242424, stop:1 #323232);\n"
"}\n"
"\n"
"QDockWidget::close-button, QDockWidget::float-button\n"
"{\n"
"    text-align: center;\n"
"    spacing: 1px; /* spacing between items in the tool bar */\n"
"    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #323232, stop: 0.5 #242424, stop:1 #323232);\n"
"}\n"
"\n"
"QDockWidget::close-button:hover, QDockWidget::float-button:hover\n"
"{\n"
"    background: #242424;\n"
"}\n"
"\n"
"QDockWidget::close-button:pressed, QDockWidget::float-button:pressed\n"
"{\n"
"    padding: 1px -1px -1px 1px;\n"
"}\n"
"\n"
"QMainWindow::separator\n"
"{\n"
"    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #161616, stop: 0.5 #151515, stop: 0.6 #212121, stop:1 #343434);\n"
"    color: white;\n"
"    padding-left: 4px;\n"
"    border: 1px solid #4c4c4c;\n"
"    spacing: 3px; /* spacing between items in the tool bar */\n"
"}\n"
"\n"
"QMainWindow::separator:hover\n"
"{\n"
"\n"
"    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #d7801a, stop:0.5 #b56c17 stop:1 #ffa02f);\n"
"    color: white;\n"
"    padding-left: 4px;\n"
"    border: 1px solid #6c6c6c;\n"
"    spacing: 3px; /* spacing between items in the tool bar */\n"
"}\n"
"\n"
"QToolBar::handle\n"
"{\n"
"     spacing: 3px; /* spacing between items in the tool bar */\n"

"}\n"
"\n"
"QMenu::separator\n"
"{\n"
"    height: 2px;\n"
"    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:0 #161616, stop: 0.5 #151515, stop: 0.6 #212121, stop:1 #343434);\n"
"    color: white;\n"
"    padding-left: 4px;\n"
"    margin-left: 10px;\n"
"    margin-right: 5px;\n"
"}\n"
"\n"
"QProgressBar\n"
"{\n"
"    border: 2px solid grey;\n"
"    border-radius: 5px;\n"
"    text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk\n"
"{\n"
"    background-color: #d7801a;\n"
"    width: 2.15px;\n"
"    margin: 0.5px;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"    color: #b1b1b1;\n"
"    border: 1px solid #444;\n"
"    border-bottom-style: none;\n"
"    background-color: #323232;\n"
"    padding-left: 10px;\n"
"    padding-right: 10px;\n"
"    padding-top: 3px;\n"
"    padding-bottom: 2px;\n"
"    margin-right: -1px;\n"
"}\n"
"\n"
"QTabWidget::pane {\n"
"    border: 1px solid #444;\n"
"    top: 1px;\n"
"}\n"
"\n"
"QTabBar::tab:last\n"
"{\n"
"    margin-right: 0; /* the last selected tab has nothing to overlap with on the right */\n"
"    border-top-right-radius: 3px;\n"
"}\n"
"\n"
"QTabBar::tab:first:!selected\n"
"{\n"
" margin-left: 0px; /* the last selected tab has nothing to overlap with on the right */\n"
"\n"
"\n"
"    border-top-left-radius: 3px;\n"
"}\n"
"\n"
"QTabBar::tab:!selected\n"
"{\n"
"    color: #b1b1b1;\n"
"    border-bottom-style: solid;\n"
"    margin-top: 3px;\n"
"    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:1 #212121, stop:.4 #343434);\n"
"}\n"
"\n"
"QTabBar::tab:selected\n"
"{\n"
"    border-top-left-radius: 3px;\n"
"    border-top-right-radius: 3px;\n"
"    margin-bottom: 0px;\n"
"}\n"
"\n"
"QTabBar::tab:!selected:hover\n"
"{\n"
"    /*border-top: 2px solid #ffaa00;\n"
"    padding-bottom: 3px;*/\n"
"    border-top-left-radius: 3px;\n"
"    border-top-right-radius: 3px;\n"
"    background-color: QLinearGradient(x1:0, y1:0, x2:0, y2:1, stop:1 #212121, stop:0.4 #343434, stop:0.2 #343434, stop:0.1 #ffaa00);\n"
"}\n"
"\n"
"QRadioButton::indicator:checked, QRadioButton::indicator:unchecked{\n"
"    color: #b1b1b1;\n"
"    background-color: #323232;\n"
"    border: 1px solid #b1b1b1;\n"
"    border-radius: 6px;\n"
"}\n"
"\n"
"QRadioButton::indicator:checked\n"
"{\n"
"    background-color: qradialgradient(\n"
"        cx: 0.5, cy: 0.5,\n"
"        fx: 0.5, fy: 0.5,\n"
"        radius: 1.0,\n"
"        stop: 0.25 #ffaa00,\n"
"        stop: 0.3 #323232\n"
"    );\n"
"}\n"
"\n"
"QCheckBox::indicator{\n"
"    color: #b1b1b1;\n"
"    background-color: #323232;\n"
"    border: 1px solid #b1b1b1;\n"
"    width: 9px;\n"
"    height: 9px;\n"
"}\n"
"\n"
"QRadioButton::indicator\n"
"{\n"
"    border-radius: 6px;\n"
"}\n"
"\n"
"QRadioButton::indicator:hover, QCheckBox::indicator:hover\n"
"{\n"
"    border: 1px solid #ffaa00;\n"
"}\n"
"\n"
"QCheckBox::indicator:checked\n"
"{\n"

"    background-color:#3D7848;\n"
"}\n"
"\n"
"QCheckBox::indicator:disabled, QRadioButton::indicator:disabled\n"
"{\n"
"    border: 1px solid #444;\n"
"}\n"
"QListWidget::item {\n"
"    color: #b1b1b1;\n"
"    border: 1px solid #444;\n"
"    border-bottom-style: none;\n"
"    background-color: #323232;\n"
"    padding-left: 10px;\n"
"    padding-right: 10px;\n"
"    padding-top: 3px;\n"
"    padding-bottom: 2px;\n"
"    margin-right: -1px;\n"
"    height: 28px;\n"
"}")    

        # Set up a main view
        self.main_Widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_Widget)

        # Initialize Widgets
        self.viewer = vidViewer.ImageViewer()
        self.viewer.frame_change_signal.connect(self.slider_on_frame_change)
        self.viewer.metadata_update_signal.connect(self.on_vid_metadata_change)
        self.menuBar = QtWidgets.QMenuBar(self)
        self.setStyleSheet("""
                                           QMenuBar {
                                           font-size:18px;
                                           background : transparent;
                                           }
                                           """)

        # Tool bar Top - Functions to analyze the current image
        self.toolBar_Top = QtWidgets.QToolBar(self)

        # Tool bar Bottom  - Play/pause buttons
        self.toolBar_Bottom = QtWidgets.QToolBar(self)

        # Frame Slider Bottom  - Easily move between frames
        self.slider_bottom = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_bottom.setMinimum(1)
        self.slider_bottom.setMaximum(100)
        self.slider_bottom.setValue(1)
        self.slider_bottom.setTickInterval(1)
        self.slider_bottom.setEnabled(False)
        self.slider_bottom.sliderPressed.connect(self.slider_pressed)
        self.slider_bottom.sliderReleased.connect(self.slider_value_final)
        self.groupBox = QtWidgets.QGroupBox(self.viewer)
        # Status Bar Bottom - Show the current frame number
        self.frameLabel = QtWidgets.QLabel('')
        self.frameLabel.setFont(QtGui.QFont("Times", 10))
        self.statusBar_Bottom = QtWidgets.QStatusBar()
        self.statusBar_Bottom.setFont(QtGui.QFont("Times", 10))
        self.statusBar_Bottom.addPermanentWidget(self.frameLabel)
        self.groupBox.setEnabled(True)
        self.groupBox.setGeometry(QtCore.QRect(1700, 40, 200, 201))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")

        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(52, 140, 94, 32))
        #self.pushButton.setFlat(True)
        self.pushButton.setObjectName("pushButton")
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_2.setGeometry(QtCore.QRect(20, 30, 161, 41))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.initUI()
        self.show()

    def initUI(self):
        # Populate the different menus
        # File Menu
        if os.path.isfile("file2.txt"):
            os.remove("file2.txt")
        file_menu = self.menuBar.addMenu("&File")

        load_video = file_menu.addAction("Load File")
        load_video.setShortcut("Ctrl+O")
        load_video.setStatusTip(
            'Load video file, accepted format : .mp4')
        load_video.triggered.connect(self.select_video_file)
        #Video Menu
        video_menu = self.menuBar.addMenu("&Video")

        play_video = video_menu.addAction("Play Video")
        play_video.setShortcut("Ctrl+P")
        play_video.setStatusTip('Play video at given playback speed')
        play_video.triggered.connect(lambda: self.viewer.play())

        stop_video = video_menu.addAction("Stop Video")
        stop_video.setShortcut("Ctrl+L")
        stop_video.setStatusTip('Stop video playback')
        stop_video.triggered.connect(lambda: self.viewer.pause())

        self.save_video = file_menu.addAction("Save File")
        self.save_video.setShortcut("Ctrl+S")
        self.save_video.setEnabled(False)
        self.save_video.triggered.connect(self.save_video_file)
        self.toolBar_Top.setIconSize(QtCore.QSize(50, 50))
        for action in self.toolBar_Top.actions():
            widget = self.toolBar_Top.widgetForAction(action)
            widget.setFixedSize(50, 50)
        self.toolBar_Top.setMinimumSize(self.toolBar_Top.sizeHint())
        self.toolBar_Top.setStyleSheet('QToolBar{spacing:8px;}')

        # Bottom toolbar population
        play_action = QtWidgets.QAction('Play', self)
        play_action.setShortcut('Shift+S')
        play_action.setIcon(QtGui.QIcon('./icons/play-arrow.png'))
        play_action.triggered.connect(lambda: self.viewer.play())

        stop_action = QtWidgets.QAction('Stop', self)
        stop_action.setShortcut('Shift+Z')
        stop_action.setIcon(QtGui.QIcon('./icons/pause.png'))
        stop_action.triggered.connect(lambda: self.viewer.pause())

        fastforward_action = QtWidgets.QAction('Jump Forward', self)
        fastforward_action.setShortcut('Shift+D')
        fastforward_action.setIcon(QtGui.QIcon('./icons/fast-forward.png'))
        fastforward_action.triggered.connect(lambda: self.viewer.jump_frames(1))

        rewind_action = QtWidgets.QAction('Jump Back', self)
        rewind_action.setShortcut('Shift+A')
        rewind_action.setIcon(QtGui.QIcon('./icons/rewind.png'))
        rewind_action.triggered.connect(lambda: self.viewer.jump_frames(-1))
        # spacer widget for left
        left_spacer = QtWidgets.QWidget(self)
        left_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Expanding)
        # spacer widget for right
        right_spacer = QtWidgets.QWidget(self)
        right_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        # fill the bottom toolbar
        self.toolBar_Bottom.addWidget(left_spacer)
        self.toolBar_Bottom.addActions(
            (rewind_action, play_action, stop_action, fastforward_action))
        self.toolBar_Bottom.addWidget(right_spacer)
        self.toolBar_Bottom.setIconSize(QtCore.QSize(35, 35))

        self.toolBar_Bottom.setMinimumSize(self.toolBar_Bottom.sizeHint())
        self.toolBar_Bottom.setStyleSheet('QToolBar{spacing:8px;}')
        self.pushButton.setText("Apply")
        self.pushButton.setEnabled(False)
        self.pushButton.clicked.connect(self.Apply_Filter)
        self.comboBox_2.setItemText(0,"modelfile")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.menuBar)
        layout.addWidget(self.toolBar_Top)
        layout.addWidget(self.viewer)
        layout.addWidget(self.toolBar_Bottom)
        layout.addWidget(self.slider_bottom)
        self.setStatusBar(self.statusBar_Bottom)
        self.main_Widget.setLayout(layout)
        self.setGeometry(600, 100, self.sizeHint().width(),
                         self.sizeHint().height())
        #f1 = open("file.txt","r")
        #print(f1.read())
        self.newindow = VideoPlayer(self)
        self.show()
    def Apply_Filter(self):
        self.statusBar_Bottom.showMessage(' DSR started : 0% ')
        if self.comboBox_2.currentIndex() == 0 :
            inputs = self.filename
            print(inputs)     
            outputs = "output.mp4"
            self.loader = model_loader(inputs,outputs)
            self.loader.report.connect(self.model_run)
            self.loader.start()
         
    #videofilename = []
    # Video Interactions
    def select_video_file(self):
        self.filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load Video File',
            '.', "Video/ files (*.avi  *.mp4)")
        self.extension = os.path.splitext(self.filename)[1]
        #self.name =  os.path.splitext(self.filename)[0]
        self.path = os.path.dirname(self.filename)
  
        self.filename = os.path.normpath(self.filename)
        file = open("file.txt","w")
        file.write(self.filename)
        #fileonly =  os.path.basename(filename)
        #print(fileonly)
        
        self.open_video_file(self.filename)
        self.statusBar_Bottom.showMessage(' Video loaded : ' + self.filename)
            #self.save_video(filename)
    def save_video_file(self):
        self.filename_new,_ = QtWidgets.QFileDialog.getSaveFileName(
        self, 'Save Video File',
        '.', "Video files(*.avi *.mp4)")
        #print("filename",self.filename_new)
        #self.path_new = os.path.dirname(self.filename_new)
        self.extension = os.path.splitext(self.filename)[1]
        print("file name",self.filename_new)
        self.extension_new = os.path.splitext(self.filename_new)[1]
        print("extension",self.extension_new)
        print("length of file name",len(self.extension_new))
        #print("extension length",len(self.extension_new))
        self.name_new =  os.path.splitext(self.filename_new)[0]
        print(self.name_new)
        self.name_new = self.name_new + self.extension
        print("changed name",self.name_new)  
        self.setWindowTitle('Deep Sign Recognition' + self.filename_new)
        self.statusBar_Bottom.showMessage('Video saving at : ' + self.filename_new)
        videoclip = VideoFileClip(self.filename)
        print(self.filename)
        shutil.copy2("output.mp4",self.filename_new)
        self.statusBar_Bottom.showMessage('Video saved ')
    def open_video_file(self, file: str) -> bool:
        try:
            vid_cap = cv2.VideoCapture(file)
        except cv2.error as e:
            return False
        self.viewer.set_reader(vid_cap)
        self.pushButton.setEnabled(True)
        return True
    # Events:
    def resizeEvent(self, event):

        self.viewer.fitInView()

    def model_run(self, val):  
        self.statusBar_Bottom.showMessage(' DSR started : ' + "{}%".format(str(int(val))))
        if int(val) == 100:
            time.sleep(5)
            self.statusBar_Bottom.showMessage(' DSR completed : ' + "{}%".format(str(int(val))))
            file = open("file2.txt","w")
            self.save_video.setEnabled(True)
            self.open_video_file("output.mp4")

    @QtCore.pyqtSlot(int, float, object)
    def on_vid_metadata_change(self, length: int, fps: int, resolution: Tuple[int, int]):
        self.frameLabel.setText('Resolution : ' + str(resolution))
        self.slider_bottom.setMinimum(1)
        self.slider_bottom.setMaximum(length)
        self.slider_bottom.setValue(1)
        self.slider_bottom.setTickInterval(1)
        self.slider_bottom.setEnabled(True)
    # Slider Interactions
    @QtCore.pyqtSlot()
    def slider_pressed(self):
        self._video_playing = self.viewer.is_playing()
        self.viewer.pause()

    @QtCore.pyqtSlot()
    def slider_value_final(self):
        frame = self.slider_bottom.value()
        self.viewer.seek_frame(frame-1)
        if self._video_playing:
            self.viewer.play()
            self._video_playing = False
    @QtCore.pyqtSlot(int)
    def slider_on_frame_change(self, frame: int):
        self.slider_bottom.setValue(frame+1)
    def closeEvent(self, event):
        if os.path.isfile("file2.txt"):
            messageBox = QMessageBox()
            title = "Quit Application?"
            message = "WARNING !!\n\nIf you quit without saving, any changes made to the file will be lost.\n\nSave file before quitting?"
            reply = messageBox.question(self, title, message, messageBox.Yes | messageBox.No |
                    messageBox.Cancel, messageBox.Cancel)
            if reply == messageBox.Yes:
                self.save_video_file()
                event.ignore()
            elif reply == messageBox.No:
                event.accept()
            else:
                event.ignore()
        else:
            reply = QMessageBox.question(self, 'Quit Application?',
                                     'Are you sure you want to quit?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                if not type(event) == bool:
                    event.accept()
                else:
                    sys.exit()
            else:
                if not type(event) == bool:
                    event.ignore()

    def invalid_path_alert_message(self):
        messageBox = QMessageBox()
        messageBox.setWindowTitle("Invalid file")
        messageBox.setText("The file name is not valid. Please use a valid file name")
        messageBox.exec()
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # freeze_support()

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    app.setStyle(QtWidgets.QStyleFactory.create('Cleanlooks'))
    GUI = MainWindow()
    app.setWindowIcon(QtGui.QIcon('icon.ico'))
    GUI.setWindowIcon(QtGui.QIcon('icon.ico'))
    # GUI.show()
    GUI.showMaximized()
    app.exec_()
