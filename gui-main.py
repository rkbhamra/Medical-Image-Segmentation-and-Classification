import sys

import PyQt6 as pqt
import PyQt6.QtWidgets as q
from PyQt6.QtGui import QPixmap, QPicture
from PyQt6.QtWidgets import QApplication, QMainWindow
import PyQt6.QtCore as qtc
import PyQt6.QtGui as qtg
import cv2
import time
import numpy as np

from classification import use_model


winWidth = 580
winHeight = 680
# winWidth = 800
# winHeight = 700
butWidth = 90
butHeight = 30
imgWidth = 512
imgHeight = 512
imgSize = (512, 512)

buttonCSS = """
            QPushButton {
                font-size: 18px;
                background-color: #404040;
                border-color: rgba(0, 0, 0, 0);
            }
            QPushButton:disabled {
                background-color: #282828;
                border-color: #202020;
            }
            QPushButton:hover {
                border-radius: 6px;
                background-color: %s;
                border-color: #101010;
            }
        """

class ImageUploadWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.setGeometry(300, 200, winWidth, winHeight)
        # cw = q.QWidget(self)
        # layout = q.QGridLayout()
        # newMsg = q.QLabel("helo", parent=self)
        # newMsg.move(20, 40)
        # subWindow = q.QWidget(parent=self)
        # subWindow.setGeometry(50, 100, 40, 40)
        # titleTextHolder = q.QLabel(self)
        # titleTextHolder.setGeometry(0, 10, winWidth, 50)

        self.contrastLevel = 0
        self.outImage = None

        titleText = q.QLabel("Tuberculosis Detection Algorithm using X-Ray", self)
        titleText.setStyleSheet("""
            QLabel {
                font-size: 18pt;
            }
            """)
        titleText.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        titleText.setGeometry(0, 5, winWidth, 50)

        self.imageInputBlock = q.QLabel(self)
        self.imageInputBlock.setGeometry(34, 60, imgWidth, imgHeight)

        self.outputText = q.QLabel("", self)
        self.outputText.setStyleSheet("""
            QLabel {
                font-size: 14pt;
            }
            """)
        self.outputText.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.outputText.setGeometry(0, 614, winWidth, 40)
        
        self.accuracyText = q.QLabel("", self)
        self.accuracyText.setStyleSheet("""
            QLabel {
                font-size: 14pt;
            }
            """)
        self.accuracyText.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.accuracyText.setGeometry(0, 642, winWidth, 40)

        self.curImgSize = None
        
        # self.imageOutputBlock = q.QLabel(self)
        # self.imageOutputBlock.setGeometry(50, 380, imgWidth, imgHeight)

        self.imagePlaceHolderTop = q.QPushButton("No Image Loaded Yet", self)
        self.imagePlaceHolderTop.setGeometry(34, 60, imgWidth, imgHeight)
        self.imagePlaceHolderTop.clicked.connect(self.importImage)

        # self.imagePlaceHolderBot = q.QPushButton("No Image Loaded Yet", self)
        # self.imagePlaceHolderBot.setGeometry(50, 380, imgWidth, imgHeight)
        # self.imagePlaceHolderBot.clicked.connect(self.saveImage)
        self.imageExportBtn = q.QPushButton("Save Image", self)
        self.imageExportBtn.setGeometry(410, 580, 130, 40)
        self.imageExportBtn.setStyleSheet(buttonCSS % ("#60b0e0"))
        self.imageExportBtn.setEnabled(False)
        self.imageExportBtn.clicked.connect(self.saveImage)

        self.ctUp = q.QPushButton("+", self)
        self.ctUp.setGeometry(130, 580, 40, 40)
        self.ctUp.setStyleSheet(buttonCSS % ("#a0a0a0"))
        self.ctUp.setEnabled(False)
        self.ctUp.clicked.connect(lambda state, inc=True: self.contrast(inc))

        self.ctText = q.QLabel("--", self)
        self.ctText.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.ctText.setGeometry(80, 580, 50, 40)
        self.ctText.setStyleSheet("""
            QLabel {
                font-size: 14pt;
            }
            """)
        
        self.ctDown = q.QPushButton("-", self)
        self.ctDown.setGeometry(40, 580, 40, 40)
        self.ctDown.setStyleSheet(buttonCSS % ("#a0a0a0"))
        self.ctDown.setEnabled(False)
        self.ctDown.clicked.connect(lambda state, inc=False: self.contrast(inc))

        ctLabel = q.QLabel("Contrast", self)
        ctLabel.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        ctLabel.setGeometry(40, 610, 130, 40)
        ctLabel.setStyleSheet("""
            QLabel {
                font-size: 12pt;
            }
            """)

        processButton = q.QPushButton("Go", self)
        processButton.setGeometry(260, 580, 60, 40)
        processButton.setStyleSheet(buttonCSS % ("#60e060"))
        processButton.clicked.connect(self.processImage)
        processButton.setToolTip("Run algorithm to check for tuberculosis\nNOTE: RIGHT NOW THIS SIMPLY PUTS THE OTHER IMAGE")


    def formatContrast(self, c):
        match (c):
            case 0:
                return "--"
            case -10:
                return "-1"
            case 10:
                return "+1"
        s = str(abs(c))
        print("STRING", s)
        return ("+" if c > 0 else "-") + "0." + s


    def contrast(self, up):
        self.contrastLevel += (1 if up else -1)
        if self.contrastLevel < -10:
            self.contrastLevel = -10
        elif self.contrastLevel > 10:
            self.contrastLevel = 10
        else:
            imgStr = self.outImage.bits().asarray(512*512*4)
            npImg = np.frombuffer(imgStr, dtype=np.uint8)
            npImg.resize((512, 512, 4))

            ctLevel = self.contrastLevel / 18 + 1
            npImg = cv2.addWeighted(npImg, ctLevel, npImg, 0, self.contrastLevel * -12.7)
            npImg = cv2.cvtColor(npImg, 5)

            npQimg = qtg.QImage(npImg.data, 512, 512, 512 * 4, qtg.QImage.Format.Format_RGBX8888)
            print(npQimg.size(), npQimg.hasAlphaChannel())
            npPix = QPixmap(npQimg)
            self.imageInputBlock.setPixmap(npPix)

            self.ctText.setText(self.formatContrast(self.contrastLevel))
    
    def importImage(self):
        fname = q.QFileDialog.getOpenFileName(
            self,
            "Open File",
            "./res/example_data",
            "PNG Files (*.png);; All Files (*)",
        )
        print(fname)
        # topButStyle = q.QStyle()
        # topButStyle
        if fname[0] != '':
            imageInput = QPixmap()
            imageInput.load(fname[0])
            self.imageInputBlock.setPixmap(imageInput.scaled(self.imageInputBlock.size()))#, qtc.Qt.AspectRatioMode.IgnoreAspectRatio))
            self.imageInputBlock.setObjectName(fname[0])
            # self.imageInputBlock.resize(imageInput.width(), imageInput.height())
            # self.imagePlaceHolderTop.setFlat(True)
            self.imagePlaceHolderTop.setText("Click to Replace Image")
            self.imagePlaceHolderTop.setStyleSheet("""
                QPushButton {
                    border-radius: 16pt;
                    background-color: rgba(0, 0, 0, 0);
                    color: transparent;
                    border-color: rgba(0, 0, 0, 0);
                    font-size: 24pt;
                }
                QPushButton:hover {
                    color: #ffffff;
                    background-color: rgba(0, 0, 0, 111);
                }
                QPushButton:pressed {
                    color: #ffffff;
                    background-color: rgba(127, 127, 127, 111);
                }
            """)
            print(imageInput.size())
            self.curImgSize = (imageInput.width(), imageInput.height())
            self.imageExportBtn.setEnabled(False)
            self.ctUp.setEnabled(False)
            self.ctDown.setEnabled(False)
            self.ctText.setText("--")
            self.contrastLevel = 0
            # print(self.imagePlaceHolderTop.styleSheet())
            # self.imagePlaceHolderTop.update()
            # print(self.imagePlaceHolderTop.styleSheet())
        return
    
    def saveImage(self):
        fname = q.QFileDialog.getSaveFileName(
            self,
            "Save File",
            "./OutputImages",
            "PNG Files (*.png);; All Files (*)",
        )
        # fname = "C:/Users/nicho/source/repos/Medical-Image-Segmentation-and-Classification/OutputImages/test1.png"
        if fname[0]:
            # with open(fname, 'w') as f:
            #     f.write(self.imageOutputBlock.pixmap().toImage())
            img = self.imageInputBlock.grab(self.imageInputBlock.rect())
            print(img)
            # self.imageOutputBlock.render()
            img.save(fname[0])
    
    def exportImage(self, filename):
        # fname = q.QFileDialog.getOpenFileName(
        #     self,
        #     "Open File",
        #     "./Images",
        #     "PNG Files (*.png);; All Files (*)",
        # )
        print(filename)
        if(filename != ''):
            imageInput = QPixmap()
            imageInput.load(filename)
            self.imageOutputBlock.setPixmap(imageInput.scaled(self.imageOutputBlock.size()))#, qtc.Qt.AspectRatioMode.IgnoreAspectRatio))
            # self.imageOutputBlock.resize(imageInput.width(), imageInput.height())
            self.imagePlaceHolderBot.setText("Click to Save Image")
            self.imagePlaceHolderBot.setStyleSheet("""
                QPushButton {
                    border-radius: 16pt;
                    background-color: rgba(0, 0, 0, 0);
                    color: transparent;
                    border-color: rgba(0, 0, 0, 0);
                    font-size: 24pt;
                }
                QPushButton:hover {
                    color: #ffffff;
                    background-color: rgba(0, 0, 0, 111);
                }
                QPushButton:pressed {
                    color: #ffffff;
                    background-color: rgba(127, 127, 127, 111);
                }
            """)
        return
    
    def processImage(self):
        currentImage = self.imageInputBlock.objectName()
        print(currentImage)
        if currentImage == "":
            return
        else:
            lung_class, acc, mask = use_model('models/tuberculosis_model.keras', currentImage)
            print(lung_class, acc, mask, mask.shape)
            lung_class = lung_class.split(" ")[0]
            lung_class = lung_class[0].upper() + lung_class[1:]
            # print(lung_class)
            self.outputText.setText("Prediction: " + lung_class)
            accStr = f'Accuracy: {acc * 100:.2f}%'
            # print(accStr)
            self.accuracyText.setText(accStr)
            # print("B", mask.shape)
            # cv2.imshow("BEFORE", mask)
            # cv2.waitKey()
            # mask = cv2.resize(mask, (self.imageInputBlock.width(), self.imageInputBlock.height()))
            # # masked_img = masked_img.scaled(self.imageInputBlock.size())
            # print("A", mask.shape)
            # cv2.imshow("AFTER", mask)
            # cv2.waitKey()
            # mask = list(reversed(mask))
            
            imgH, imgW, _ = mask.shape
            # mask2 = np.require(mask, np.uint8, 'C')
            masked_img = qtg.QImage(mask.data, imgW, imgH, imgW * 3, qtg.QImage.Format.Format_RGB888)
            # masked_img = QPixmap()
            # mask2 = cv2.Mat
            # masked_img.loadFromData(mask.data)
            # masked_img.
            mask_px = QPixmap(masked_img)
            
            # mask_px.convertFromImage(masked_img)
            # mask2 = int(mask * 256)
            # mask_px.loadFromData(mask2.data)
            # print(mask2)
            # maskmap = mask_px.mask().
            # print(mask_px.)
            self.imageInputBlock.setPixmap(mask_px.scaled(self.imageInputBlock.size()))#, qtc.Qt.AspectRatioMode.IgnoreAspectRatio))
            self.imageExportBtn.setEnabled(True)
            self.ctUp.setEnabled(True)
            self.ctDown.setEnabled(True)
            self.outImage = self.imageInputBlock.pixmap().toImage()
            # self.imageInputBlock.setObjectName(masked_img[0])
            # if currentImage[-5] == '2':
            #     self.exportImage("C:/Users/nicho/source/repos/Medical-Image-Segmentation-and-Classification/InputImages/sample_image.png")
            # else:
            #     self.exportImage("C:/Users/nicho/source/repos/Medical-Image-Segmentation-and-Classification/InputImages/sample_image_2.png")
        return
        


def main():
    app = QApplication([])
    mainWindow = ImageUploadWindow()    
    mainWindow.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


