import cv2
import functions as fs

IM_TMP = 'set/notHealthy/8e04135b-2279-43a7-af41-147d96395478___FREC_Scab 3143.JPG'

img = cv2.imread(IM_TMP)
segmented_img = fs.c_segmenting(img)

marked_img = fs.marking(segmented_img)

cv2.imwrite('results/image8.png',marked_img)

cv2.imshow('seg_img', marked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


