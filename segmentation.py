import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

# Major algorithm
def identify_lungs(binary_image):
    lung_image = np.zeros_like(binary_image)
    
    height, width = binary_image.shape
    mid = width // 2  

    for y in range(height):
        row = binary_image[y]
        
        # Left lung
        start_idx = None
        for x in range(mid):
            if row[x] == 255:
                if start_idx is None:
                    start_idx = x-2
            elif row[x] == 0 and start_idx is not None:
                lung_image[y, start_idx:x] = 255
                start_idx = None

        # Right lung
        start_idx = None
        for x in range(mid, width):
            if row[x] == 255:
                if start_idx is None:
                    start_idx = x-5
            elif row[x] == 0 and start_idx is not None:
                lung_image[y, start_idx:x] = 255
                start_idx = None

    return lung_image

def segmentation(img_path, size):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    _, width = img.shape

    # white padding to the top
    white_padding = np.zeros((50, width))
    white_padding[:, :] = 255
    img = np.row_stack((white_padding, img))

    # invert image colors (binary format with white as lung region)
    img = 255 - img

    img[img > 80] = 255  # Keep this threshold unchanged
    img[img <= 80] = 0

    # black padding to the bottom
    black_padding = np.zeros((50, width))
    img = np.row_stack((black_padding, img))

    # Apply morphological closing to fill small holes
    kernel = np.ones((12, 12), np.uint8)
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    closing = np.uint8(closing)

    lung_identified_image = cv.flip(identify_lungs(cv.flip(identify_lungs(cv.resize(closing,(size,size))),1)),1)

    contours, _ = cv.findContours(lung_identified_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv.contourArea, reverse=True)[:2]

    lung_mask = np.zeros_like(lung_identified_image)

    cv.drawContours(lung_mask, contours, -1, (255), thickness=cv.FILLED)

    img = cv.resize(img, (size, size))
    for i in range(lung_mask.shape[0]):
        for j in range(lung_mask.shape[1]):
            if lung_mask[i][j] == 0:
                img[i][j] = 0

    return lung_mask, img

def calculate_white_pixel_count(img_path):
    lung_mask, _ = segmentation(img_path, 512)
    white_pixel_count = np.sum(lung_mask == 255)  # Count white pixels (255)
    return white_pixel_count

if __name__ == '__main__':
    white_pixel_counts = []  # List to store white pixel counts

    for img in os.listdir(r'C:\Users\reetr\OneDrive\Desktop\Medical-Image-Segmentation-and-Classification\CXRpng-train\img'):
        img_path = os.path.join(r'C:\Users\reetr\OneDrive\Desktop\Medical-Image-Segmentation-and-Classification\CXRpng-train\img', img)
        white_pixel_count = calculate_white_pixel_count(img_path)
        white_pixel_counts.append(white_pixel_count)
        print(f'{img}: {white_pixel_count} white pixels')

    # Plotting the frequency of the number of white pixels in the whole list
    plt.figure(figsize=(10, 5))
    plt.hist(white_pixel_counts, bins=20, edgecolor='black')
    plt.title('Distribution of White Pixels in Lung Masks')
    plt.xlabel('Number of White Pixels')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
