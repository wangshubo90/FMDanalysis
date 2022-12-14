# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\wangs\dev\Ultrasond\FMD.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DialogRef(object):
    def setupUi(self, DialogRef):
        DialogRef.setObjectName("DialogRef")
        DialogRef.resize(474, 430)
        self.widget = QtWidgets.QWidget(DialogRef)
        self.widget.setGeometry(QtCore.QRect(50, 31, 382, 364))
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.labelRefDir = QtWidgets.QLabel(self.widget)
        self.labelRefDir.setMinimumSize(QtCore.QSize(120, 0))
        self.labelRefDir.setObjectName("labelRefDir")
        self.horizontalLayout_2.addWidget(self.labelRefDir)
        self.showVideoFile = QtWidgets.QLineEdit(self.widget)
        self.showVideoFile.setObjectName("showVideoFile")
        self.horizontalLayout_2.addWidget(self.showVideoFile)
        self.browseVideo = QtWidgets.QPushButton(self.widget)
        self.browseVideo.setObjectName("browseVideo")
        self.horizontalLayout_2.addWidget(self.browseVideo)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.labelTarDir = QtWidgets.QLabel(self.widget)
        self.labelTarDir.setMinimumSize(QtCore.QSize(120, 0))
        self.labelTarDir.setObjectName("labelTarDir")
        self.horizontalLayout.addWidget(self.labelTarDir)
        self.showOutput = QtWidgets.QLineEdit(self.widget)
        self.showOutput.setObjectName("showOutput")
        self.horizontalLayout.addWidget(self.showOutput)
        self.browseOutput = QtWidgets.QPushButton(self.widget)
        self.browseOutput.setObjectName("browseOutput")
        self.horizontalLayout.addWidget(self.browseOutput)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.minBox = QtWidgets.QLineEdit(self.widget)
        self.minBox.setText("")
        self.minBox.setObjectName("minBox")
        self.gridLayout.addWidget(self.minBox, 2, 1, 1, 1)
        self.labelRefDir_4 = QtWidgets.QLabel(self.widget)
        self.labelRefDir_4.setObjectName("labelRefDir_4")
        self.gridLayout.addWidget(self.labelRefDir_4, 0, 2, 1, 1)
        self.freqBox = QtWidgets.QLineEdit(self.widget)
        self.freqBox.setObjectName("freqBox")
        self.gridLayout.addWidget(self.freqBox, 0, 1, 1, 1)
        self.labelMin = QtWidgets.QLabel(self.widget)
        self.labelMin.setObjectName("labelMin")
        self.gridLayout.addWidget(self.labelMin, 2, 0, 1, 1)
        self.labelRefDir_2 = QtWidgets.QLabel(self.widget)
        self.labelRefDir_2.setObjectName("labelRefDir_2")
        self.gridLayout.addWidget(self.labelRefDir_2, 0, 0, 1, 1)
        self.fpsBox = QtWidgets.QLineEdit(self.widget)
        self.fpsBox.setObjectName("fpsBox")
        self.gridLayout.addWidget(self.fpsBox, 1, 1, 1, 1)
        self.freqBox_2 = QtWidgets.QLineEdit(self.widget)
        self.freqBox_2.setObjectName("freqBox_2")
        self.gridLayout.addWidget(self.freqBox_2, 0, 3, 1, 1)
        self.labelRefDir_3 = QtWidgets.QLabel(self.widget)
        self.labelRefDir_3.setObjectName("labelRefDir_3")
        self.gridLayout.addWidget(self.labelRefDir_3, 1, 0, 1, 1)
        self.labelMax = QtWidgets.QLabel(self.widget)
        self.labelMax.setObjectName("labelMax")
        self.gridLayout.addWidget(self.labelMax, 2, 2, 1, 1)
        self.maxBox = QtWidgets.QLineEdit(self.widget)
        self.maxBox.setObjectName("maxBox")
        self.gridLayout.addWidget(self.maxBox, 2, 3, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.saveButtonVessel = QtWidgets.QPushButton(self.widget)
        self.saveButtonVessel.setObjectName("saveButtonVessel")
        self.gridLayout_2.addWidget(self.saveButtonVessel, 2, 1, 1, 1)
        self.loadrefButton = QtWidgets.QPushButton(self.widget)
        self.loadrefButton.setObjectName("loadrefButton")
        self.gridLayout_2.addWidget(self.loadrefButton, 3, 0, 1, 1)
        self.saveButton = QtWidgets.QPushButton(self.widget)
        self.saveButton.setObjectName("saveButton")
        self.gridLayout_2.addWidget(self.saveButton, 3, 1, 1, 1)
        self.calcThickness = QtWidgets.QPushButton(self.widget)
        self.calcThickness.setObjectName("calcThickness")
        self.gridLayout_2.addWidget(self.calcThickness, 4, 0, 1, 2)
        self.cropButtonVideo = QtWidgets.QPushButton(self.widget)
        self.cropButtonVideo.setObjectName("cropButtonVideo")
        self.gridLayout_2.addWidget(self.cropButtonVideo, 2, 0, 1, 1)
        self.readButton = QtWidgets.QPushButton(self.widget)
        self.readButton.setObjectName("readButton")
        self.gridLayout_2.addWidget(self.readButton, 0, 0, 1, 1)
        self.browseConfig = QtWidgets.QPushButton(self.widget)
        self.browseConfig.setObjectName("browseConfig")
        self.gridLayout_2.addWidget(self.browseConfig, 0, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_2)
        self.labelLog = QtWidgets.QLabel(self.widget)
        self.labelLog.setMinimumSize(QtCore.QSize(0, 100))
        self.labelLog.setFrameShape(QtWidgets.QFrame.Box)
        self.labelLog.setWordWrap(True)
        self.labelLog.setObjectName("labelLog")
        self.verticalLayout_2.addWidget(self.labelLog)

        self.retranslateUi(DialogRef)
        QtCore.QMetaObject.connectSlotsByName(DialogRef)

    def retranslateUi(self, DialogRef):
        _translate = QtCore.QCoreApplication.translate
        DialogRef.setWindowTitle(_translate("DialogRef", "Dialog"))
        self.labelRefDir.setText(_translate("DialogRef", "Selet Video File"))
        self.browseVideo.setText(_translate("DialogRef", "1. Browse"))
        self.labelTarDir.setText(_translate("DialogRef", "Selet Output Directory"))
        self.browseOutput.setText(_translate("DialogRef", "2. Browse"))
        self.labelRefDir_4.setText(_translate("DialogRef", "Calc Freq"))
        self.labelMin.setText(_translate("DialogRef", "MinEdge"))
        self.labelRefDir_2.setText(_translate("DialogRef", "Save Freq"))
        self.labelRefDir_3.setText(_translate("DialogRef", "FPS"))
        self.labelMax.setText(_translate("DialogRef", "MaxEdge"))
        self.saveButtonVessel.setText(_translate("DialogRef", "5. Calibrate"))
        self.loadrefButton.setText(_translate("DialogRef", "6. Load Reference Image"))
        self.saveButton.setText(_translate("DialogRef", "Save raw imgs"))
        self.calcThickness.setText(_translate("DialogRef", "7. Calculate Thickness"))
        self.cropButtonVideo.setText(_translate("DialogRef", "4. Crop Video -- Vessel"))
        self.readButton.setText(_translate("DialogRef", "3. Read"))
        self.browseConfig.setText(_translate("DialogRef", "Load Config (optional??? skip to 6)"))
        self.labelLog.setText(_translate("DialogRef", "Welcome!"))
