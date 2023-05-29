import cv2
import functions as fs

IM_TMP = 'set/notHealthy/7b3e4313-9abb-4105-8be9-2020f1101e58___FREC_Scab 3441_90deg.JPG'

img = cv2.imread(IM_TMP)
segmented_img = fs.c_segmenting(img)
cv2.imshow('seg_img', segmented_img)

segmented_img_gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
img_contours, hierarchy = cv2.findContours(segmented_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(segmented_img, img_contours, -1, (0, 0, 255), 3)

marked_img = fs.marking(segmented_img)

marked_img_gray = cv2.cvtColor(marked_img, cv2.COLOR_BGR2GRAY)
marked_img_contours, hierarchy = cv2.findContours(marked_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(segmented_img, marked_img_contours, -1, (0, 0, 255), 3)

#cv2.imwrite('D:\PythonProjectsFolder\ImageProcessingProject\results', segmented_img)

cv2.imshow('seg_img', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


