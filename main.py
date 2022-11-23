import sys
import os
import time
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QInputDialog
from PyQt5.uic import loadUi
import cv2
import numpy as np
import fitVessel
from fitVessel import *
from crop import video_to_image
from crop import click_event
from scipy import signal
import json
from matplotlib import pyplot as plt

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
        self.saveButton.clicked.connect(self.saveCroppedImage)
        self.loadrefButton.clicked.connect(self.loadRefImage)
        self.browseConfig.clicked.connect(self.loadConfig)
        self.save_freq = 100
        self.freqBox.setText(str(self.save_freq))
        self.freqBox.textChanged[str].connect(self._updateSavFreq)
        self.fpsBox.setText("23")
        self.fps = int(self.fpsBox.text())
        self.dt = 1 / self.fps
        self.fpsBox.textChanged[str].connect(self._updateFPS)
        self.minBox.setText(str(fitVessel.CANNY_LOWER))
        fitVessel.CANNY_LOWER=float(self.minBox.text())
        self.minBox.textChanged[str].connect(self._updateThreshold)
        self.maxBox.setText(str(fitVessel.CANNY_HIGHER))
        fitVessel.CANNY_HIGHER=float(self.maxBox.text())
        self.maxBox.textChanged[str].connect(self._updateThreshold)
        self.videofile = ""
        self.outputDir = ""
        self.vidcap=None
        self.firstImage = None
        self.imshowName = 'image'
        self._hasbox = False
        self.refimg=None
        self.scaling=1.0
        self.sampling_freq = 1
        self.freqBox_2.setText(str(self.sampling_freq))
        self.freqBox_2.textChanged[str].connect(self._updateSampling)
        # global vesselBoxLeft, vesselBoxRight, vesselBoxUp, vesselBoxDown , signalBoxLeft, signalBoxRight, signalBoxUp, signalBoxDown
    
    def browseVideoFile(self):
        filename = QFileDialog.getOpenFileName(None, 'Load Video', '', 'Video Files (*.avi *.mp4 *.mov)')[0]
        self.videofile = filename
        self.showVideoFile.setText(filename)
    
    def _updateThreshold(self):
        self.labelLog.clear()
        if self.minBox.text() != "":
            fitVessel.CANNY_LOWER=float(self.minBox.text())
            self.labelLog.setText(f"MinEdge: {fitVessel.CANNY_LOWER} MaxEdge: {fitVessel.CANNY_HIGHER}")
            
        if self.maxBox.text() != "":
            fitVessel.CANNY_HIGHER=float(self.maxBox.text())
            self.labelLog.setText(f"MinEdge: {fitVessel.CANNY_LOWER} MaxEdge: {fitVessel.CANNY_HIGHER}")
    
    def _updateFPS(self):
        if self.fpsBox.text() != "":
            self.fps = int(self.fpsBox.text())
        
    def _updateSampling(self):
        if self.freqBox_2.text() != "":
            self.sampling_freq = int(self.freqBox_2.text()) 
        
    def _updateSavFreq(self):
        if self.freqBox.text() != "":
            self.save_freq = int(self.freqBox.text()) 
        
    def browseOutputDir(self):
        dirname=QFileDialog.getExistingDirectory(self, "Choose Output Directory", r"C:\Users\wangs\dev\Ultrasond\output1")
        self.outputDir = dirname
        self.showOutput.setText(self.outputDir)
        
    def loadRefImage(self):
        filename = QFileDialog.getOpenFileName(None, 'Load Reference Image', '', 'Video Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)')[0]
        if filename:
            self.refimg = cv2.imread(filename)
            # self.refimg = cv2.cvtColor(self.refimg, cv2.COLOR_BGR2GRAY)
            
    def loadConfig(self):
        filename = QFileDialog.getOpenFileName(None, 'Load Configuration', '', 'JSON Files (*.json)')[0]
        if filename:
            config = json.load(open(filename, "r"))
            for k ,v in config.items():
                setattr(self, k, v)
                
            self.fpsBox.setText(str(config.get("fps", str(self.fps))))
            self.freqBox.setText(str(config.get("save_freq", str(self.save_freq))))
            self.freqBox_2.setText(str(config.get("sampling_freq", str(self.sampling_freq))))
            self._hasbox=True
            fitVessel.CANNY_LOWER = self.MinEdge
            fitVessel.CANNY_HIGHER = self.MaxEdge
            self.minBox.setText(str(fitVessel.CANNY_LOWER))
            self.maxBox.setText(str(fitVessel.CANNY_HIGHER))
            self.labelLog.clear()
            self.labelLog.setText(json.dumps(config))
            

    def read(self):
        self.labelLog.setText(f"Loading Video: {self.videofile}")
        try:
            self.vidcap = cv2.VideoCapture(self.videofile)
            self.fps = int(self.fpsBox.text())
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
    
    def saveCroppedImage(self):
        if not self._hasbox:
            self.labelLog.setText(f"Please draw a box first!")
            return
        else:
            self.vidcap = cv2.VideoCapture(self.videofile)
            # freq = int(self.freqBox.text())
            video_to_image(self.vidcap, 
                           self.vesselBoxLeft, 
                           self.vesselBoxRight, 
                           self.vesselBoxUp, 
                           self.vesselBoxDown, 
                           fps=23,
                           freq = self.save_freq,
                           dstdir=self.outputDir)
    
    def calibrate_scale_vessel(self):
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
                    self._hasbox = False
                elif event == cv2.EVENT_LBUTTONUP:
                    # displaying the coordinates
                    box.append((x, y))
                    self._draw_markers(self.firstImage.copy(), box)     
                    self._hasbox = True
                    scale, done = QInputDialog.getDouble(self, "Calibration", "Enter real length (mm)")
                    boxarray = np.array(box)
                    # pixel_length = np.sqrt(np.sum(np.square(boxarray[0] - boxarray[1])))
                    pixel_length = abs(boxarray[0,1] - boxarray[1, 1])
                    self.scaling = scale / pixel_length
                    self.labelLog.setText(str(f"Scaling factor: {self.scaling:3.2f} mm/pixel"))
            
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
                    self._hasbox = False
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
                    self._hasbox = True
            
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
        if not self._hasbox:
            self.labelLog.setText(f"Please draw a box first!")
            return
        elif not os.path.exists(self.videofile):
            self.labelLog.setText(f"Video file does not exist: {self.videofile}")
            return
        elif not os.path.exists(self.outputDir):
            self.labelLog.setText(f"Output directory does not exist: {self.outputDir}")
            return
        else:
            # save configuration
            config = {
                "sampling_freq": self.sampling_freq,
                "scaling": self.scaling,
                "save_freq": self.save_freq,
                "fps": self.fps,
                "vesselBoxLeft": self.vesselBoxLeft,
                "vesselBoxRight": self.vesselBoxRight,
                "vesselBoxUp": self.vesselBoxUp,
                "vesselBoxDown": self.vesselBoxDown,
                "MinEdge": fitVessel.CANNY_LOWER,
                "MaxEdge": fitVessel.CANNY_HIGHER,
            }
            json.dump(config, open(os.path.join(self.outputDir, "config.json"), "w"), indent=4)
            # process first frame
            i = 0
            curtime = int(i*self.dt)
            self.vidcap = cv2.VideoCapture(self.videofile)
            success, self.firstImage = self.vidcap.read()
            image = self._crop(self.firstImage)
            guess, x_grid, image, edge1, edge2 = fit(image, plot=False, init_func=polynomial_init_condition, edge_func=calc_edge_polynomial, fit_func=fit_vessel, refimg=self.refimg)
            image = cv2.polylines(image,[np.array([x_grid, edge1]).T.astype(np.int32)], False, (255,0,0))
            image = cv2.polylines(image,[np.array([x_grid, edge2]).T.astype(np.int32)], False, (255,0,0))
            cv2.imwrite(os.path.join(self.outputDir, "{:d}m{:d}s.png".format(curtime//60, curtime%60)), image)
            thickness = calc_thickness(guess, x_grid, edge_func=calc_edge_polynomial)* self.scaling
            results = [[0, thickness]]
            t1 = time.time()
            while success:
                i += 1
                # read next frame
                success, image = self.vidcap.read()
                if not image is None:
                    if i%self.sampling_freq==0:
                        try:
                            self.labelLog.clear()
                            self.labelLog.setText("In process: {:d}m{:d}".format(curtime//60, curtime%60))
                            image = self._crop(image)
                            # image = cv2.imread(os.path.join(self.outputDir, "orgCropped", "{:d}m{:d}s.png".format(curtime//60, curtime%60)))
                            guess, x_grid, image, edge1, edge2 = fit(image, plot=False, init_func=polynomial_init_condition, edge_func=calc_edge_polynomial, fit_func=fit_vessel, refimg=self.refimg)
                            if i % self.save_freq==0:
                                image = cv2.polylines(image,[np.array([x_grid, edge1]).T.astype(np.int32)], False, (255,0,0))
                                image = cv2.polylines(image,[np.array([x_grid, edge2]).T.astype(np.int32)], False, (255,0,0))
                                curtime = int(i*self.dt)
                                cv2.imwrite(os.path.join(self.outputDir, "{:d}m{:d}s.png".format(curtime//60, curtime%60)), image)
                
                            thickness = calc_thickness(guess, x_grid, edge_func=calc_edge_polynomial)* self.scaling
                            results.append([i*self.dt, thickness])
                        except Exception as ex:
                            self.labelLog.setText(str(ex))
                else:
                    break
            
            results = np.array(results)
            b, a = signal.butter(2, 1/(self.fps*30/(np.log(self.sampling_freq**2)+1)), "low", analog = False) 
            col_filtered = signal.filtfilt(b, a, results[:, 1])
            # col_conv = np.convolve(col_filtered, np.ones(69), 'same') / 69
            results = np.concatenate([results, np.expand_dims(col_filtered, axis=1)], axis=1)
            
            try:
                np.savetxt(os.path.join(self.outputDir, "results.csv"), results, delimiter=",", header="Time(s),Raw(mm),Butterworth,Conv1d")
            except Exception:
                np.savetxt(os.path.join(self.outputDir, "results_.csv"), results, delimiter=",")
            t2 = time.time()
            t = t2 - t1
            self.labelLog.setText(f"Finished. Used {t//60} mins and {t%60:.2f} seconds")
            plt.plot(results[:, 0], results[:, 1], label="raw")
            plt.plot(results[:, 0], results[:, 2], label="filtered")
            plt.show()
            
            return
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui=resource_path("FMD.ui")
    mainWindow = MainWindow(ui)
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(mainWindow)
    widget.setFixedHeight(430)
    widget.setFixedWidth(470)
    widget.show()
    sys.exit(app.exec_())