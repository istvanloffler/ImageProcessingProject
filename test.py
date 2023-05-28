import cv2
import numpy as np
from matplotlib import pyplot as plt

IM_TMP = 'pics_base/Apple-rust/0a41c25a-f9a6-4c34-8e5c-7f89a6ac4c40___FREC_C.Rust 9807.JPG'
IM_TMP = 'pics_base/Apple-scrab/6b5e1187-9b7b-4b4c-912f-b2175619f35f___FREC_Scab 3121_270deg.JPG'
IM_TMP = 'pics_base/Apple-scrab/1d9d67e2-5603-4710-ae2b-6cb0b922ae61___FREC_Scab 3122_270deg.JPG'
IM_TMP = 'pics_base/Apple-scrab/1a41bab0-45e0-4dda-a798-9bf4a998f1b6___FREC_Scab 3450_90deg.JPG'
IM_TMP = 'pics_base/Apple-scrab/0c620ec5-11cf-4120-94ab-1311e99df147___FREC_Scab 3131_270deg.JPG'

#kmeans
img = cv2.imread(IM_TMP)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

twoDimage = img.reshape((-1,3))
twoDimage = np.float32(twoDimage)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
K = 6
attempts = 10

ret, label, center = cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape(img.shape)

cv2.imshow('', result_image)
cv2.waitKey(0)


img2 = cv2.imread(IM_TMP)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
cv2.resize(img2,(256,256))
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
edges = cv2.dilate(cv2.Canny(thresh,0,255),None)
cnt = sorted(cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2],key=cv2.contourArea)[-2]
mask = np.zeros((256,256), np.uint8)
masked = cv2.drawContours(mask,[cnt],-1,255,-1)
dst = cv2.bitwise_and(img2,img2,mask=mask)
segmented = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
cv2.imshow('',segmented)
cv2.waitKey(0)


img3 = cv2.imread(IM_TMP)
#img3 = cv2.GaussianBlur(img3, (3, 3), 0)
hsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)

# lower bound and upper bound for Green color
lower_bound = np.array([20, 30, 0])
upper_bound = np.array([100, 255, 255])

# find the colors within the boundaries
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# define kernel size
kernel = np.ones((7, 7), np.uint8)

# Remove unnecessary noise from mask
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

segmented_img = cv2.bitwise_and(img3, img3, mask=mask)

cv2.imshow('',segmented_img)
cv2.waitKey(0)
