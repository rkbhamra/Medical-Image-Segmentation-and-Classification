# Lungs have Tuberculosis

Lungs have Tuberculosis
This project was made as a part of the CPS843 course at TMU, Toronto, Canada. The model aims to review x-ray images and find out if the person has TB or not. It cannot detect any other chest diseases yet.
Segmentation was based off: https://homepages.inf.ed.ac.uk/rbf/BOOKS/BANDB/LIB/bandb4_3.pdf
#Installation Guide:
1. pip install -r requirements.txt

Segmentation
Gaussian Blur
Detect Edges
Link Edges
Get lungs area
Take that mask, overlap it with the actual image, set everything that is not lung mask to 0.
