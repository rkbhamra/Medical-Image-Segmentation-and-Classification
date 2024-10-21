import sys

import PyQt6.QtWidgets as q
from PyQt6.QtGui import QPixmap, QPicture
from PyQt6.QtWidgets import QApplication, QMainWindow
import PyQt6.QtCore as qtc


winWidth = 580
winHeight = 680
# winWidth = 800
# winHeight = 700
butWidth = 90
butHeight = 30
imgWidth = 480
imgHeight = 270
imgSize = (480, 270)

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

        titleText = q.QLabel("Tuberculosis Detection Algorithm using X-Ray", self)
        titleText.setStyleSheet("""
            QLabel {
                font-size: 18pt;
            }
            """)
        titleText.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        titleText.setGeometry(0, 5, winWidth, 50)

        self.imageInputBlock = q.QLabel(self)
        self.imageInputBlock.setGeometry(50, 60, imgWidth, imgHeight)
        
        self.imageOutputBlock = q.QLabel(self)
        self.imageOutputBlock.setGeometry(50, 380, imgWidth, imgHeight)

        # self.imagePlaceHolderTop = q.QLabel("No Image Loaded Yet", self)
        # self.imagePlaceHolderTop.mouseReleaseEvent=self.importImage
        self.imagePlaceHolderTop = q.QPushButton("No Image Loaded Yet", self)
        self.imagePlaceHolderTop.setGeometry(50, 60, imgWidth, imgHeight)
        # self.imagePlaceHolderTop.setObjectName("imgTop")
        self.imagePlaceHolderBot = q.QPushButton("No Image Loaded Yet", self)
        self.imagePlaceHolderBot.setGeometry(50, 380, imgWidth, imgHeight)

        aaa = self.imagePlaceHolderTop.focusWidget()
        print(aaa)
        # print(self.imagePlaceHolderTop.styleSheet())

        processButton = q.QPushButton("Go", self)
        processButton.setGeometry(260, 340, 60, 30)
        processButton.clicked.connect(self.processImage)
        processButton.setToolTip("Run algorithm to check for tuberculosis\nNOTE: RIGHT NOW THIS SIMPLY PUTS THE OTHER IMAGE")



        
        # addImageButton = q.QPushButton("Import Image", self)
        # addImageButton.setGeometry(700, 100, butWidth, butHeight)
        # addImageButton.clicked.connect(self.importImage)
        # addImageButton2 = q.QPushButton("Export Image", self)
        # addImageButton2.setGeometry(700, 150, butWidth, butHeight)
        # addImageButton2.clicked.connect(self.exportImage)

        # layout.addWidget(self.imageInputBlock, 0, 0)
        # layout.addWidget(self.imageOutputBlock, 1, 0)
        # layout.addWidget(addImageButton, 1, 0)
        # cw.setLayout(layout)
        # addImageButton.click(self.importImage)
        # self.resize(imageInput.width(), imageInput.height())
        self.imagePlaceHolderTop.clicked.connect(self.importImage)
        self.imagePlaceHolderBot.clicked.connect(self.exportImage)

    
    def importImage(self, event):
        fname = q.QFileDialog.getOpenFileName(
            self,
            "Open File",
            "./Images",
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
            # print(self.imagePlaceHolderTop.styleSheet())
            # self.imagePlaceHolderTop.update()
            # print(self.imagePlaceHolderTop.styleSheet())
        return
    
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
            self.imagePlaceHolderBot.setText("Click to Replace Image")
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
            if currentImage[-5] == '2':
                self.exportImage("C:/Users/nicho/source/repos/Medical-Image-Segmentation-and-Classification/Images/sample_image.png")
            else:
                self.exportImage("C:/Users/nicho/source/repos/Medical-Image-Segmentation-and-Classification/Images/sample_image_2.png")
        return
        


def main():
    app = QApplication([])
    mainWindow = ImageUploadWindow()    
    mainWindow.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


