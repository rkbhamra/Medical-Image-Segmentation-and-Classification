import sys

import PyQt6.QtWidgets as q
from PyQt6.QtGui import QPixmap, QPicture
from PyQt6.QtWidgets import QApplication, QMainWindow
import PyQt6.QtCore as qtc


winWidth = 800
winHeight = 700
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
        self.imageInputBlock = q.QLabel(self)
        self.imageInputBlock.setGeometry(50, 50, imgWidth, imgHeight)
        
        self.imageOutputBlock = q.QLabel(self)
        self.imageOutputBlock.setGeometry(50, 350, imgWidth, imgHeight)

        self.imagePlaceHolderTop = q.QPushButton("No Image Loaded Yet", self)
        self.imagePlaceHolderTop.setGeometry(50, 50, imgWidth, imgHeight)
        self.imagePlaceHolderTop.setObjectName("imgTop")
        # self.imagePlaceHolderTop.setStyleSheet("""
        #     QPushButton { background-color: blue; }
        #     QPushButton:hover { background-color: red; }
        #     """)
        # self.imagePlaceHolderTop.setFocusPolicy()
        # self.imagePlaceHolderTop.setEnabled(False)
        # self.imagePlaceHolderTop.setFlat(False)
        # self.imagePlaceHolderTop.setWindowOpacity(0.5)
        test = q.QLabel("CLICK", self)
        test.setGeometry(300, 50, 100, 100)
        test.setStyleSheet("""
            QWidget { background-color: blue; }
            QWidget:hover { background-color: red; }
            """)
        # test.clicked(self.importImage)
        
        self.imagePlaceHolderBot = q.QPushButton("No Image Loaded Yet", self)
        self.imagePlaceHolderBot.setGeometry(50, 350, imgWidth, imgHeight)
        # self.imagePlaceHolderBot.setDefault(False)



        
        addImageButton = q.QPushButton("Import Image", self)
        addImageButton.setGeometry(700, 100, butWidth, butHeight)
        addImageButton.clicked.connect(self.importImage)
        addImageButton2 = q.QPushButton("Export Image", self)
        addImageButton2.setGeometry(700, 150, butWidth, butHeight)
        addImageButton2.clicked.connect(self.exportImage)
        # layout.addWidget(self.imageInputBlock, 0, 0)
        # layout.addWidget(self.imageOutputBlock, 1, 0)
        # layout.addWidget(addImageButton, 1, 0)
        # cw.setLayout(layout)
        # addImageButton.click(self.importImage)
        # self.resize(imageInput.width(), imageInput.height())

    
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
        imageInput = QPixmap()
        imageInput.load(fname[0])
        self.imageInputBlock.setPixmap(imageInput.scaled(self.imageInputBlock.size()))#, qtc.Qt.AspectRatioMode.IgnoreAspectRatio))
        # self.imageInputBlock.resize(imageInput.width(), imageInput.height())
        self.imagePlaceHolderTop.setFlat(True)
        self.imagePlaceHolderTop.setText("Click to Replace Image")
        self.imagePlaceHolderTop.setStyleSheet("""
            QPushButton#imgTop {
                background-color: blue;
                color: transparent;
            }
            QPushButton#imgTop:hover {
                opacity: 1;
                color: #df0000;
                background-color: red;
                font-size: 24pt;
            }
        """)
        return
    
    def exportImage(self):
        fname = q.QFileDialog.getOpenFileName(
            self,
            "Open File",
            "./Images",
            "PNG Files (*.png);; All Files (*)",
        )
        print(fname)
        imageInput = QPixmap()
        imageInput.load(fname[0])
        self.imageOutputBlock.setPixmap(imageInput.scaled(self.imageOutputBlock.size()))#, qtc.Qt.AspectRatioMode.IgnoreAspectRatio))
        # self.imageOutputBlock.resize(imageInput.width(), imageInput.height())
        return
    
    # def eventFilter(self, a0, a1):
    #     return super().eventFilter(a0, a1)
        


def main():
    app = QApplication([])
    mainWindow = ImageUploadWindow()    
    mainWindow.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


