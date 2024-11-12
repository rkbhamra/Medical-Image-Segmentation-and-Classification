import numpy as np
import cv2 as cv
import scipy.ndimage as nd
from matplotlib import pyplot as plt

img = cv.imread(r'C:\Users\reetr\OneDrive\Desktop\CPS843\Medical-Image-Segmentation-and-Classification\res\example_data\img\CHNCXR_0025_0.png', cv.IMREAD_GRAYSCALE)

#blurred = cv.GaussianBlur(img, (5, 5), 0)
edges = cv.Canny(img, 30, 50)
plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.show()

fill_im = nd.binary_fill_holes(edges).astype(np.uint8)
plt.imshow(fill_im, cmap='gray')
plt.title('Region Filling')
plt.show()

#elevation
elevation_map = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
plt.imshow(elevation_map, cmap='gray')
plt.title('Elevation Map')
plt.show()

#markers
markers = np.zeros_like(img, dtype=np.int32)
markers[img < 30] = 1  # Background
markers[img > 150] = 2  # Foreground

plt.imshow(markers, cmap='jet')
plt.title('Markers')
plt.show()

#segmentation
segmentation = cv.watershed(cv.cvtColor(img, cv.COLOR_GRAY2BGR), markers)


plt.imshow(segmentation, cmap='gray')
plt.title('Segmentation')
plt.show()
