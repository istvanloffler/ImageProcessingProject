import cv2
import numpy as np
import matplotlib.pyplot as plt


def BF_FeatureMatcher(des1, des2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    no_of_matches = brute_force.match(des1, des2)

    # finding the humming distance of matches
    no_of_matches = sorted(no_of_matches, key=lambda x:x.distance)
    return no_of_matches


def detector(image1, image2):

    detect = cv2.ORB_create()

    key_point1, descrip1 = detect.detectAndCompute(image1, None)
    key_point2, descrip2 = detect.detectAndCompute(image2, None)
    return (key_point1,descrip1,key_point2,descrip2)


def read_image(path1, path2):
    read_img1 = cv2.imread(path1)
    read_img2 = cv2.imread(path2)
    return (read_img1,read_img2)


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
    upper_bound = np.array([100, 200, 200])

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # define kernel size
    kernel = np.ones((3, 3), np.uint8)

    # Remove unnecessary noise from mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    invert = cv2.bitwise_not(mask)

    segmented_img = cv2.bitwise_and(pic, pic, mask=invert)

    return segmented_img


def c_segmenting(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower bound and upper bound for Green color
    lower_bound = np.array([20, 30, 0])
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
