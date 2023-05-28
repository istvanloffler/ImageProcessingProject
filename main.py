import cv2
import numpy as np
import functions as fs

img1 = cv2.imread('pics_test/Apple-scab/0c620ec5-11cf-4120-94ab-1311e99df147___FREC_Scab 3131_270deg.JPG')
img2 = cv2.imread('pics_base/Crp/0b4a52e3-e15e-4117-b2e8-7cdb5dca3ce9___FREC_Scab 3137.JPG')
img3 = cv2.imread('pics_base/Apple-rust/0a41c25a-f9a6-4c34-8e5c-7f89a6ac4c40___FREC_C.Rust 9807.JPG')
img4 = cv2.imread('pics_base/Apple-scrab/8e04135b-2279-43a7-af41-147d96395478___FREC_Scab 3143.JPG')

marked1 = fs.c_segmenting(img1)
marked2 = fs.c_segmenting(img3)

gray_pic1, gray_pic2 = fs.convert_grayscale(marked1, marked2)

key_pt1, descrip1, key_pt2, descrip2 = fs.detector(gray_pic1, gray_pic2)

number_of_matches = fs.BF_FeatureMatcher(descrip1, descrip2)
num1 = len(number_of_matches)

fs.display_output(gray_pic1, key_pt1, gray_pic2, key_pt2, number_of_matches)

cv2.waitKey(0)
cv2.destroyAllWindows()


