import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread(r'C:\Users\reetr\OneDrive\Desktop\CPS843\Medical-Image-Segmentation-and-Classification\res\example_data\img\CHNCXR_0025_0.png', cv.IMREAD_GRAYSCALE)
edges = cv.Canny(img,0,50)
 
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
plt.show()
