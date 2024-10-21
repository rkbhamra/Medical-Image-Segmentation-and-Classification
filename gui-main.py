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
        # titleTextHolder.setStyleSheet("""
        #     QLabel {
        #         font-size: 18pt;
        #         align-self: center;
        #         align-content: center;
        #         align-items: center;
        #     }
        #     """)
        titleText.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        titleText.setGeometry(0, 10, winWidth, 50)

        self.imageInputBlock = q.QLabel(self)
        self.imageInputBlock.setGeometry(50, 100, imgWidth, imgHeight)
        
        self.imageOutputBlock = q.QLabel(self)
        self.imageOutputBlock.setGeometry(50, 400, imgWidth, imgHeight)

        # self.imagePlaceHolderTop = q.QLabel("No Image Loaded Yet", self)
        # self.imagePlaceHolderTop.mouseReleaseEvent=self.importImage
        self.imagePlaceHolderTop = q.QPushButton("No Image Loaded Yet", self)
        self.imagePlaceHolderTop.setGeometry(50, 100, imgWidth, imgHeight)
        # self.imagePlaceHolderTop.setObjectName("imgTop")
        # self.imagePlaceHolderTop.setStyleSheet("""
        #     QPushButton::before { 
        #         content: "No Image Yet"; 
        #     }
        #     QPushButton:hover::before {
        #         content: "Click to Load Image" !important; 
        #     }
        #     QPushButton:hover {
        #         border-radius: 2pt;
        #         background-color: #882288;
        #     }
        #     """)
        # self.imagePlaceHolderTop.setFocusPolicy()
        # self.imagePlaceHolderTop.setEnabled(False)
        # self.imagePlaceHolderTop.setFlat(False)
        # self.imagePlaceHolderTop.setWindowOpacity(0.5)
        test = q.QLabel("CLICK", self)
        test.setGeometry(500, 50, 100, 100)
        test.setStyleSheet("""
            QWidget { background-color: blue; }
            QWidget:hover { background-color: red; }
            """)
        # test.mouseReleaseEvent=self.importImage
        # test.clicked(self.importImage)
        
        self.imagePlaceHolderBot = q.QPushButton("No Image Loaded Yet", self)
        self.imagePlaceHolderBot.setGeometry(50, 400, imgWidth, imgHeight)
        # self.imagePlaceHolderBot.setDefault(False)

        aaa = self.imagePlaceHolderTop.focusWidget()
        print(aaa)
        # print(self.imagePlaceHolderTop.styleSheet())



        
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
    
    def exportImage(self):
        fname = q.QFileDialog.getOpenFileName(
            self,
            "Open File",
            "./Images",
            "PNG Files (*.png);; All Files (*)",
        )
        print(fname)
        if(fname[0] != ''):
            imageInput = QPixmap()
            imageInput.load(fname[0])
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
    
    # def eventFilter(self, a0, a1):
    #     return super().eventFilter(a0, a1)
        


def main():
    app = QApplication([])
    mainWindow = ImageUploadWindow()    
    mainWindow.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


