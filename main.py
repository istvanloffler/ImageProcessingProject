import cv2
import functions as fs

IM_TMP = 'set/notHealthy/0b4a52e3-e15e-4117-b2e8-7cdb5dca3ce9___FREC_Scab 3137_90deg.JPG'
#IM_TMP = 'pics_base/Apple-healthy/2b0d21b2-2320-4cb4-8d40-4684f0c91e55___RS_HL 7505.JPG'

img = cv2.imread(IM_TMP)
segmented_img = fs.c_segmenting_g(img)
cv2.imshow('seg_img', segmented_img)

segmented_img_gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
img_contours, hierarchy = cv2.findContours(segmented_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(segmented_img, img_contours, -1, (0, 0, 255), 3)

marked_img = fs.marking(segmented_img)

marked_img_gray = cv2.cvtColor(marked_img, cv2.COLOR_BGR2GRAY)
marked_img_contours, hierarchy = cv2.findContours(marked_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(segmented_img, marked_img_contours, -1, (0, 0, 255), 3)

cv2.imshow('seg_img', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


