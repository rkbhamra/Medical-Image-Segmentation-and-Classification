import numpy as np
import cv2 as cv

# Load and preprocess the image
img = cv.imread(r'C:\Users\reetr\OneDrive\Desktop\CPS843\Medical-Image-Segmentation-and-Classification\res\example_data\img\CHNCXR_0025_0.png', cv.IMREAD_GRAYSCALE)
_, thresholded_img = cv.threshold(img, 70, 255, cv.THRESH_BINARY)
inverted_img = cv.bitwise_not(thresholded_img)

# Detect edges and find contours
edges = cv.Canny(inverted_img, 50, 150, apertureSize=3)
contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create a blank mask for drawing selected contours
mask = np.zeros_like(img, dtype=np.uint8)

for cnt in contours:
    # Approximate the contour for smoothness
    epsilon = 0.01 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    
    # Filter based on area (adjust values as needed)
    area = cv.contourArea(cnt)
    if area < 5000 or area > 50000:
        continue  # Skip small or excessively large contours

    # Get bounding box and filter based on aspect ratio (e.g., lungs are taller than they are wide)
    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = h / w
    if aspect_ratio < 1.5:  # Adjust based on lung shape in your images
        continue  # Skip contours that don’t match expected lung shape

    # Additional filtering based on position, if necessary
    if y > img.shape[0] // 2:  # Skip contours located in the lower half
        continue

    # If contour passed all checks, draw it on the mask
    cv.drawContours(mask, [approx], -1, 255, thickness=cv.FILLED)

# Use the mask to isolate lung regions on the original image
highlighted_lungs = cv.bitwise_and(img, img, mask=mask)

# Optional: Display or save the results
resized_img = cv.resize(img, (512, 512))
resized_mask = cv.resize(mask, (512, 512))
resized_highlighted_lungs = cv.resize(highlighted_lungs, (512, 512))

cv.imshow("Original Image", resized_img)
cv.imshow("Filtered Lung Mask", resized_mask)
cv.imshow("Highlighted Lungs", resized_highlighted_lungs)
cv.waitKey(0)
cv.destroyAllWindows()
