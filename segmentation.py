import numpy as np
import cv2 as cv


img = cv.imread(r'C:\Users\reetr\OneDrive\Desktop\CPS843\Medical-Image-Segmentation-and-Classification\res\example_data\img\CHNCXR_0025_0.png', cv.IMREAD_GRAYSCALE)
height, width = img.shape

white_padding = np.zeros((50, width))
white_padding[:, :] = 255
img = np.row_stack((white_padding, img))
img = 255 - img
img[img > 70] = 255
img[img <= 70] = 0
black_padding = np.zeros((50, width))
img = np.row_stack((black_padding, img))

kernel = np.ones((10, 10), np.uint8)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
closing = np.uint8(closing)
edges = cv.Canny(closing, 70, 200)

contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


cv.imshow("Filled Lung Regions", cv.resize(cv.dilate(edges, np.ones((40, 40), np.uint8)), (512, 512)))
output_dir = r'C:\Users\reetr\OneDrive\Desktop\CPS843\Medical-Image-Segmentation-and-Classification\output'
cv.imwrite(f'{output_dir}/filled_lung_regions.png', closing)

cv.waitKey()
cv.destroyAllWindows()
