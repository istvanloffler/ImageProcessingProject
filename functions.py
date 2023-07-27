import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_grayscale(img1, img2):
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return (gray_img1,gray_img2)


def display_output(pic1, kpt1, pic2, kpt2, best_match):
    output_image = cv2.drawMatches(pic1,kpt1,pic2,kpt2,best_match[:15],None,flags=2)
    plt.imshow(output_image)
    plt.show()


def marking(pic):
    # convert to hsv colorspace
    hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)

    # lower bound and upper bound for Green color
    lower_bound = np.array([40, 30, 0])
    upper_bound = np.array([100, 255, 255])

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # define kernel size
    kernel = np.ones((3, 3), np.uint8)

    # Remove unnecessary noise from mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    invert = cv2.bitwise_not(mask)

    segmented_img = cv2.bitwise_and(pic, pic, mask=invert)
    marked_img_gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    img_contours, hierarchy = cv2.findContours(marked_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pic, img_contours, -1, (0, 0, 255), 3)

    return pic


def c_segmenting(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower bound and upper bound for Green color
    lower_bound = np.array([30, 30, 40])
    upper_bound = np.array([100, 255, 255])

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    # define kernel size
    kernel = np.ones((7, 7), np.uint8)

    # Remove unnecessary noise from mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    segmented_img = cv2.bitwise_and(img, img, mask=mask)

    return segmented_img
