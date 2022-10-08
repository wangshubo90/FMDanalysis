import sys
import os
import time
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QInputDialog
from PyQt5.uic import loadUi
import cv2
import numpy as np
from fitVessel import *

from crop import click_event

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class MainWindow(QDialog):
    def __init__(self,ui):
        super(MainWindow, self).__init__()
        loadUi(ui, self)
        self.browseVideo.clicked.connect(self.browseVideoFile)
        self.readButton.clicked.connect(self.read)
        self.cropButtonVideo.clicked.connect(self.cropVessel)
        self.browseOutput.clicked.connect(self.browseOutputDir)
        self.saveButtonVessel.clicked.connect(self.calibrate_scale_vessel)
        self.calcThickness.clicked.connect(self.fit_vessel)
        self.videofile = ""
        self.outputDir = ""
        self.vidcap=None
        self.firstImage = None
        self.imshowName = 'image'
        # global vesselBoxLeft, vesselBoxRight, vesselBoxUp, vesselBoxDown , signalBoxLeft, signalBoxRight, signalBoxUp, signalBoxDown
    
    def browseVideoFile(self):
        filename = QFileDialog.getOpenFileName(None, 'Load Video', '', 'Video Files (*.avi *.mp4)')[0]
        self.videofile = filename
        self.showVideoFile.setText(filename)
        
    def browseOutputDir(self):
        dirname=QFileDialog.getExistingDirectory(self, "Choose Output Directory")
        self.outputDir = dirname
        self.showOutput.setText(self.outputDir)

    def read(self):
        self.labelLog.setText(f"Loading Video: {self.videofile}")
        try:
            self.vidcap = cv2.VideoCapture(self.videofile)
            self.fps = 23
            self.dt = 1 / self.fps
            self.n_frames = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.labelLog.clear()
            self.labelLog.setText("Video File: {} \n\
                                  Video FPS: {:4.3f} \n\
                                  Video Time Interval: {:4.3f} \n\
                                  Video Total Frames: {:d}".format(os.path.basename(self.videofile), self.fps, self.dt, self.n_frames))
        except Exception as ex:
            # self.labelLog.clear()
            # pass
            self.labelLog.setText(str(ex))
            # self.labelLog.setText(f"Choose a valid video file")
        _, self.firstImage = self.vidcap.read()
        
    def setCropBoxEvenFunction(self, parnames):
        
        return click_event
    
    def _draw_markers(self, image, box):
        for coord in box:
            cv2.drawMarker(image, coord, color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=1)
        cv2.imshow(self.imshowName, image)
    
    def calibrate_scale_vessel(self):
        scale, done = QInputDialog.getDouble(self, "Calibration", "Enter real length (mm)")
    
    def cropVessel(self):
        if isinstance(self.firstImage, np.ndarray):
            img = self.firstImage.copy()
            cv2.imshow(self.imshowName, img)
            box = []
            def click_event(event, x, y, flags, params):
                # checking for left mouse clicks
                if event == cv2.EVENT_LBUTTONDOWN:
                    if len(box) ==0:
                        box.append((x, y))
                    else:
                        box.clear()
                        box.append((x, y))
                    # displaying the coordinates
                    self._draw_markers(self.firstImage.copy(), box)
                elif event == cv2.EVENT_LBUTTONUP:
                    # displaying the coordinates
                    
                    box.append((x, y))
                    self._draw_markers(self.firstImage.copy(), box)     
                    
                    x , y = zip(*box)
                    self.vesselBoxLeft  = min(x)
                    self.vesselBoxRight = max(x)
                    self.vesselBoxUp = min(y)
                    self.vesselBoxDown = max(y)
                    infor = 'Box Position: '+f'({self.vesselBoxLeft}, {self.vesselBoxUp}) '+f'({self.vesselBoxRight}, {self.vesselBoxDown})'
                    self.labelLog.setText(infor)
            
            cv2.setMouseCallback(self.imshowName, click_event)
            # wait for a key to be pressed to exit
            cv2.waitKey(0)
            # close the window
            try:
                cv2.destroyWindow(self.imshowName)
            except Exception:
                pass
        else:
            self.labelLog.setText("Please read a video file first!")
    
    def cropSignal(self):
        pass
    
    def run():
        pass
    
    def _crop(self, image):
        return image[self.vesselBoxUp:self.vesselBoxDown, self.vesselBoxLeft:self.vesselBoxRight]
    
    def fit_vessel(self):
        if not os.path.exists(self.videofile):
            self.showVideoFile.setText(f"Video file does not exist: {self.videofile}")
            return
        elif not os.path.exists(self.outputDir):
            self.showVideoFile.setText(f"Output directory does not exist: {self.outputDir}")
            return
        else:
            self.vidcap = cv2.VideoCapture(self.videofile)
            success, self.firstImage = self.vidcap.read()
            i = 0
            image = self._crop(self.firstImage)
            guess, x_grid, image, edge1, edge2 = fit(image.copy(), plot=False)
            image = cv2.polylines(image,[np.array([x_grid, edge1]).T.astype(np.int32)], False, (255,0,0))
            image = cv2.polylines(image,[np.array([x_grid, edge2]).T.astype(np.int32)], False, (255,0,0))
            curtime = int(i*self.dt)
            cv2.imwrite(os.path.join(self.outputDir, "{:d}m{:d}s.png".format(curtime//60, curtime%60)), image)

            thickness = calc_thickness(guess, x_grid)
            results = [[0, thickness]]
            t1 = time.time()
            while success:
                i += 1
                # read next frame
                success, image = self.vidcap.read()
                if not image is None:
                    try:
                        image = self._crop(image)
                        guess, x_grid, image, edge1, edge2 = fit(image.copy(), plot=False, guess = guess)
                        if i % 100==0:
                            image = cv2.polylines(image,[np.array([x_grid, edge1]).T.astype(np.int32)], False, (255,0,0))
                            image = cv2.polylines(image,[np.array([x_grid, edge2]).T.astype(np.int32)], False, (255,0,0))
                            curtime = int(i*self.dt)
                            cv2.imwrite(os.path.join(self.outputDir, "{:d}m{:d}s.png".format(curtime//60, curtime%60)), image)
            
                        thickness = calc_thickness(guess, x_grid)
                        results.append([i*self.dt, thickness])
                    except Exception as ex:
                        self.labelLog.setText(str(ex))
                else:
                    break
            try:
                np.savetxt(os.path.join(self.outputDir, "results.csv"), np.array(results), delimiter=",")
            except Exception:
                np.savetxt(os.path.join(self.outputDir, "results_.csv"), np.array(results), delimiter=",")
            t2 = time.time()
            t = t2 - t1
            self.labelLog.setText(f"Finished. Used {t//60} mins and {t%60} seconds")
            
            return
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui=resource_path("FMD.ui")
    mainWindow = MainWindow(ui)
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(mainWindow)
    widget.setFixedHeight(280)
    widget.setFixedWidth(600)
    widget.show()
    sys.exit(app.exec_())