from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QMainWindow, QMessageBox
from PyQt5.QtGui import QPixmap
import numpy as np
import cv2
import os
import sys

import torch
from numpy import *
import torch.nn as nn
from PIL import Image,ImageFilter
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms




class Ui_MainWindow(object):
    xrayPath = ''


    def xrayUpload(self):
            fname, _filter = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image File", '.', "(*.png *jpeg)")
            self.xrayPath = fname
            if self.xrayPath == '':
                    (self.label_12.setStyleSheet("background-color: red;"))
            else:
                    self.label_12.setStyleSheet("background-color: rgb(126, 153, 254);color: rgb(255, 255, 255);")
                    self.inputLabel.setPixmap(QPixmap(self.xrayPath))
                    self.preLabel.setPixmap(QPixmap(self.xrayPath))
                    self.orginalLabel.setPixmap(QPixmap(self.xrayPath))
                    img = cv2.imread(self.xrayPath)
                    cv2.imwrite('Original_Pic.jpeg', img)
    def xrayDeleteAllImages(self):
            if self.xrayPath == '':
                    ((self.label_9.setStyleSheet("background-color:red;")))
            else:
                #if os.path.isfile("Original_Pic.jpg"):
                os.remove("Original_Pic.jpeg")
                self.xrayPath = ''
                self.orginalLabel.setText(" ")
                self.predictLabel.setText(" ")
                self.inputLabel.setText(" ")
                self.preLabel.setText(" ")
                self.label_12.setStyleSheet("background-color: rgb(126, 153, 254);color: rgb(255, 255, 255);")
    def xraySetHeightAndWidth(self):
            if self.xrayPath == '':
                    ((self.label_15.setStyleSheet("background-color:red;")))
            else:
                    self.label_15.setStyleSheet("background-color: #85ff77;")
                    self.resizeButton.setStyleSheet("background-color: #85ff77;")
                    img = cv2.imread(self.xrayPath)
                    res = cv2.resize(img, (512, 512))
                    cv2.imwrite(self.xrayPath, res)
                    self.preLabel.setPixmap(QPixmap(self.xrayPath))
    def xrayEnhacementFunction(self):
        if self.xrayPath == '':
            self.label_15.setStyleSheet("background-color:red;")
        else:
            self.label_15.setStyleSheet("background-color: #85ff77;")
            self.enhancementButton.setStyleSheet("background-color: #85ff77;")
            img = cv2.imread(self.xrayPath, cv2.IMREAD_COLOR)
            norm_img1 = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            norm_img2 = (255 * norm_img1).astype(np.uint8)
            cv2.imwrite(self.xrayPath, norm_img2)
            self.preLabel.setPixmap(QPixmap(self.xrayPath))
            #self.gray.setPixmap(QPixmap(self.xrayPath))
    def xrayGrayGaussian(self):
            if self.xrayPath == '':
                    self.label_15.setStyleSheet("background-color:red;")
            else:
                    self.label_15.setStyleSheet("background-color: #85ff77;")
                    self.gaussianButton.setStyleSheet("background-color: #85ff77;")
                    image = Image.open(self.xrayPath)
                    image = image.filter(ImageFilter.GaussianBlur)
                    #image.show()
                    self.preLabel.setPixmap(QPixmap(self.xrayPath))
    def xrayClearPreProcessing(self):
            # self.label_8.setStyleSheet("background-color: #bca18d;\n""color: rgb(234, 255, 255);")
           self.label_15.setStyleSheet("background-color: rgb(239, 132, 58);\n""color: rgb(234, 255, 255);")
           self.preLabel.setText(" ")
    def xrayPredictionFunction(self):
        if self.xrayPath == '':
            self.label_14.setStyleSheet("background-color:red;")
        else:
            self.resultLabel.setPixmap(QPixmap(self.xrayPath))
            self.label_14.setStyleSheet("background-color: #85ff77;")
            classes = ('Covid', 'Normal', 'Viral_Pneumonia')
            transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            def image_loader(image_name):
                """load image, returns tensor"""
                image = Image.open(image_name).convert('RGB')  # load single image
                image = transform(image).float()  # apply transformation
                image = Variable(image, requires_grad=True)  # Convert it to tensor
                image = image.unsqueeze(0)
                return image

            class CNN(nn.Module):
                def __init__(self):
                    super(CNN, self).__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3)
                    self.conv2 = nn.Conv2d(32, 64, 4)
                    self.pool = nn.MaxPool2d(2, 2)
                    self.fc1 = nn.Linear(64 * 14 * 14, 32)
                    self.fc2 = nn.Linear(32, 3)

                # self.softmax = nn.Softmax(dim=1)
                def forward(self, x):
                    x = self.pool(F.relu(self.conv1(x)))
                    x = self.pool(F.relu(self.conv2(x)))
                    x = x.view(-1, 64 * 14 * 14)
                    x = self.fc1(x)
                    x = F.relu(x)
                    x = self.fc2(x)
                    # x = self.softmax(x)
                    return x

            model = CNN()
            # print(model)
            state_dict = torch.load('Modal/xrayClassifier.pth')['state_dict']
            model.load_state_dict(state_dict)
            # change this path according to your iamge path
            image = image_loader("Original_Pic.jpeg")
            output = model(image)
            print(output.data.cpu().numpy()) #HIGHEST CONFIDENCE
            _, predicted = torch.max(output, 1)
            w = classes[predicted]
            # w =str(w)

            self.predictLabel.setText(w)

    def xrayClearPredictionAndClassifictaion(self):
        self.predictLabel.setText(" ")
        self.orginalLabel.setText(" ")
        self.resultLabel.setText(" ")
        self.label_14.setStyleSheet("background-color: rgb(126, 153, 254);")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(762, 714)
        MainWindow.setMinimumSize(QtCore.QSize(762, 714))
        MainWindow.setMaximumSize(QtCore.QSize(762, 714))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.headerLabel = QtWidgets.QLabel(self.centralwidget)
        self.headerLabel.setGeometry(QtCore.QRect(0, 0, 771, 41))
        self.headerLabel.setStyleSheet("background-color: rgb(239, 132, 58);")
        self.headerLabel.setText("")
        self.headerLabel.setObjectName("headerLabel")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 210, 771, 511))
        self.tabWidget.setStyleSheet("QWidget {background-color: #f9ddd2; color: black;}QTabBar::tab:selected{background-color: rgb(239, 132, 58)}QTabBar::tab { height: 40px;font: bold,; text-align:center;}\n"
"")
        self.tabWidget.setObjectName("tabWidget")
        self.main = QtWidgets.QWidget()
        self.main.setObjectName("main")
        self.label_2 = QtWidgets.QLabel(self.main)
        self.label_2.setGeometry(QtCore.QRect(-1, 5, 761, 471))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("fypImages/roziachest-01.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.label_5 = QtWidgets.QLabel(self.main)
        self.label_5.setGeometry(QtCore.QRect(390, 30, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("background-color: rgb(239, 132, 58);\n"
"color: rgb(255, 255, 255);\n"
"border-radius:8px;")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.uploadButton = QtWidgets.QPushButton(self.main)
        self.uploadButton.setGeometry(QtCore.QRect(40, 180, 121, 31))
        self.uploadButton.setStyleSheet("border-radius:8px;")
        self.uploadButton.setObjectName("uploadButton")
        self.uploadButton.clicked.connect(self.xrayUpload)
        self.deleteButton = QtWidgets.QPushButton(self.main)
        self.deleteButton.setGeometry(QtCore.QRect(40, 230, 121, 31))
        self.deleteButton.setStyleSheet("border-radius:8px;")
        self.deleteButton.setObjectName("deleteButton")
        self.deleteButton.clicked.connect(self.xrayDeleteAllImages)
        self.inputLabel = QtWidgets.QLabel(self.main)
        self.inputLabel.setGeometry(QtCore.QRect(340, 140, 311, 271))
        self.inputLabel.setStyleSheet("background:transparent;")
        self.inputLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.inputLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.inputLabel.setLineWidth(1)
        self.inputLabel.setScaledContents(True)
        self.inputLabel.setText("")
        self.inputLabel.setObjectName("inputLabel")
        self.label_12 = QtWidgets.QLabel(self.main)
        self.label_12.setGeometry(QtCore.QRect(340, 110, 311, 31))
        self.label_12.setStyleSheet("background-color: rgb(126, 153, 254);\n"
"color: rgb(255, 255, 255);")
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.tabWidget.addTab(self.main, "Main Windows                                         ")
        self.pre = QtWidgets.QWidget()
        self.pre.setObjectName("pre")
        self.label_3 = QtWidgets.QLabel(self.pre)
        self.label_3.setGeometry(QtCore.QRect(-1, 5, 761, 471))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("fypImages/roziachest-02.png"))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")
        self.label_6 = QtWidgets.QLabel(self.pre)
        self.label_6.setGeometry(QtCore.QRect(390, 30, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("background-color: rgb(239, 132, 58);\n"
"color: rgb(255, 255, 255);\n"
"border-radius:8px;")
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.resizeButton = QtWidgets.QPushButton(self.pre)
        self.resizeButton.setGeometry(QtCore.QRect(40, 170, 121, 31))
        self.resizeButton.setStyleSheet("border-radius:8px;")
        self.resizeButton.setObjectName("resizeButton")
        self.resizeButton.clicked.connect(self.xraySetHeightAndWidth)
        self.enhancementButton = QtWidgets.QPushButton(self.pre)
        self.enhancementButton .setGeometry(QtCore.QRect(40, 220, 121, 31))
        self.enhancementButton .setStyleSheet("border-radius:8px;")
        self.enhancementButton .setObjectName("enhancementButton")
        self.enhancementButton.clicked.connect(self.xrayEnhacementFunction)
        self.gaussianButton = QtWidgets.QPushButton(self.pre)
        self.gaussianButton.setGeometry(QtCore.QRect(40, 270, 121, 31))
        self.gaussianButton.setStyleSheet("border-radius:8px;")
        self.gaussianButton.setObjectName("enhancementButton")
        self.gaussianButton.clicked.connect(self.xrayGrayGaussian)
        self.preClearButton = QtWidgets.QPushButton(self.pre)
        self.preClearButton.setGeometry(QtCore.QRect(40, 320, 121, 31))
        self.preClearButton.setStyleSheet("border-radius:8px;")
        self.preClearButton.setObjectName("medianButton")
        self.preClearButton.clicked.connect(self.xrayClearPreProcessing)
        #self.gaussianButton = QtWidgets.QPushButton(self.pre)
        #self.gaussianButton.setGeometry(QtCore.QRect(40, 320, 121, 31))
        #self.gaussianButton.setStyleSheet("border-radius:8px;")
        #self.gaussianButton.setObjectName("medianButton")
        #self.gaussianButton.clicked.connect(self.xrayGrayGaussian)
        #self.preClearButton = QtWidgets.QPushButton(self.pre)
        #self.preClearButton.setGeometry(QtCore.QRect(40, 370, 121, 31))
        #self.preClearButton.setStyleSheet("border-radius:8px;")
        #self.preClearButton.clicked.connect(self.xrayClearPreProcessing)
        self.preLabel = QtWidgets.QLabel(self.pre)
        self.preLabel.setGeometry(QtCore.QRect(340, 140, 311, 271))
        self.preLabel.setStyleSheet("background:transparent;")
        self.preLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.preLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.preLabel.setScaledContents(True)
        self.preLabel.setLineWidth(1)
        self.preLabel.setText("")
        self.preLabel.setObjectName("preLabel")
        self.label_15 = QtWidgets.QLabel(self.pre)
        self.label_15.setGeometry(QtCore.QRect(340, 110, 311, 31))
        self.label_15.setStyleSheet("background-color: rgb(126, 153, 254);\n"
"color: rgb(255, 255, 255);")
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.tabWidget.addTab(self.pre, "")
        self.clasi = QtWidgets.QWidget()
        self.clasi.setObjectName("clasi")
        self.label_4 = QtWidgets.QLabel(self.clasi)
        self.label_4.setGeometry(QtCore.QRect(-1, 5, 761, 471))
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("fypImages/roziachest-03.png"))
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName("label_4")
        self.label_7 = QtWidgets.QLabel(self.clasi)
        self.label_7.setGeometry(QtCore.QRect(390, 30, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("background-color: rgb(239, 132, 58);\n"
"color: rgb(255, 255, 255);\n"
"border-radius:8px;")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.predictionButton = QtWidgets.QPushButton(self.clasi)
        self.predictionButton.setGeometry(QtCore.QRect(30, 180, 121, 31))
        self.predictionButton.setStyleSheet("border-radius:8px;")
        self.predictionButton.setObjectName("predictionButton")
        self.predictionButton.clicked.connect(self.xrayPredictionFunction)
        self.classificationButton = QtWidgets.QPushButton(self.clasi)
        self.classificationButton.setGeometry(QtCore.QRect(30, 230, 121, 31))
        self.classificationButton.setStyleSheet("border-radius:8px;")
        self.classificationButton.setObjectName("classificationButton")
        self.classificationButton.clicked.connect(self.xrayClearPredictionAndClassifictaion)
        self.orginalLabel = QtWidgets.QLabel(self.clasi)
        self.orginalLabel.setGeometry(QtCore.QRect(290, 160, 201, 181))
        self.orginalLabel.setStyleSheet("background:transparent;")
        self.orginalLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.orginalLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.orginalLabel.setScaledContents(True)
        self.orginalLabel.setLineWidth(1)
        self.orginalLabel.setText("")
        self.orginalLabel.setObjectName("orginalLabel")
        self.resultLabel = QtWidgets.QLabel(self.clasi)
        self.resultLabel.setGeometry(QtCore.QRect(500, 160, 191, 181))
        self.resultLabel.setStyleSheet("background:transparent;")
        self.resultLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.resultLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.resultLabel.setScaledContents(True)
        self.resultLabel.setLineWidth(1)
        self.resultLabel.setText("")
        self.resultLabel.setObjectName("resultLabel")
        self.label_13 = QtWidgets.QLabel(self.clasi)
        self.label_13.setGeometry(QtCore.QRect(290, 130, 201, 31))
        self.label_13.setStyleSheet("background-color: rgb(126, 153, 254);\n"
"color: rgb(255, 255, 255);")
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.clasi)
        self.label_14.setGeometry(QtCore.QRect(500, 130, 191, 31))
        self.label_14.setStyleSheet("background-color: rgb(126, 153, 254);\n"
"color: rgb(255, 255, 255);")
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.predictLabel = QtWidgets.QLabel(self.clasi)
        self.predictLabel.setGeometry(QtCore.QRect(410, 360, 281, 35))
        self.predictLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.predictLabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.predictLabel.setText("")
        self.predictLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.predictLabel.setObjectName("predictLabel")
        self.label_8 = QtWidgets.QLabel(self.clasi)
        self.label_8.setGeometry(QtCore.QRect(290, 360, 121, 35))
        self.label_8.setFrameShape(QtWidgets.QFrame.Box)
        self.label_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.tabWidget.addTab(self.clasi, "")
        self.headerImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.headerImageLabel.setGeometry(QtCore.QRect(-1, 35, 761, 181))
        self.headerImageLabel.setText("")
        self.headerImageLabel.setPixmap(QtGui.QPixmap("fypImages/WhatsApp Image 2022-02-04 at 3.08.54 PM (1).png"))
        self.headerImageLabel.setScaledContents(True)
        self.headerImageLabel.setObjectName("headerImageLabel")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_5.setText(_translate("MainWindow", "Main Window"))
        self.uploadButton.setText(_translate("MainWindow", "Upload Image"))
        self.deleteButton.setText(_translate("MainWindow", "Delete Image"))
        self.label_12.setText(_translate("MainWindow", "Input Image"))
        self.label_6.setText(_translate("MainWindow", "Pre-Processing"))
        self.resizeButton.setText(_translate("MainWindow", "Resize"))
        self.enhancementButton .setText(_translate("MainWindow", "Enhancement"))
        self.gaussianButton.setText(_translate("MainWindow", "Gaussian Filter"))
        self.preClearButton.setText(_translate("MainWindow", "Clear All"))
        #self.gaussianButton.setText(_translate("MainWindow", "Gaussian Filter"))
        #self.preClearButton.setText(_translate("MainWindow", "Clear All"))
        self.label_15.setText(_translate("MainWindow", "Pre-Processed Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.main), _translate("MainWindow", "                     Main Windows                      "))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.pre), _translate("MainWindow", "                     Pre-Processing                     "))
        self.label_7.setText(_translate("MainWindow", "Classification"))
        self.predictionButton.setText(_translate("MainWindow", "Prediction"))
        self.classificationButton.setText(_translate("MainWindow", "Clear All"))
        self.label_13.setText(_translate("MainWindow", "Orignial Images"))
        self.label_14.setText(_translate("MainWindow", "Result Image"))
        self.label_8.setText(_translate("MainWindow", "Predict Result"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.clasi), _translate("MainWindow", "                    Classification                   "))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())