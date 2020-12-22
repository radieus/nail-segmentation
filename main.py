import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
import random as rng
from scipy.interpolate import splprep, splev

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt

images_dir = 'nails_segmentation/images/'
images_names = [f for f in listdir(images_dir) if isfile(join(images_dir, f))]

# some particular images
img_path1 = 'nails_segmentation/images/1eecab90-1a92-43a7-b952-0204384e1fae.jpg'
img_path2 = 'nails_segmentation/images/2C29D473-CCB4-458C-926B-99D0042161E6.jpg'
img_path3 = 'nails_segmentation/images/2c376c66-9823-4874-869e-1e7f5c54ec7b.jpg'
img_path4 = 'nails_segmentation/images/4c4a0dd6-e402-11e8-97db-0242ac1c0002.jpg'
img_path5 = 'nails_segmentation/images/d60a5f06-db67-11e8-9658-0242ac1c0002.jpg'
img_path6 = 'nails_segmentation/images/bf93c2e2-7b5f-4108-ae85-4ef68564d418.jpg'
img_path7 = 'nails_segmentation/images/3493127D-7B19-4E50-94AE-2401BD2A91C8.jpg'
img_path8 = 'nails_segmentation/images/09aefeec-e05f-11e8-87a6-0242ac1c0002.jpg'
img_path9 = 'nails_segmentation/images/4252e46c-e40f-4543-91ab-031917d46c5c.jpg'
img_path10 = 'nails_segmentation/images/da236f3a-8c82-4c64-9a7d-9b950fd8b47e.jpg'
img_path11 = 'nails_segmentation/images/5fad3947-76d7-4352-9329-4a92f898dd59.jpg'
img_path12 = 'nails_segmentation/images/54108996-6DA8-48F9-93DF-7ABB92F64E03.jpg'
img_path13 = 'nails_segmentation/images/feb2c029-b89c-4ce5-b208-db2114516a40.jpg'
img_path14 = 'nails_segmentation/images/865a1e90-7ad2-4ceb-b2a1-50b07875c5c7.jpg'
img_path15 = 'nails_segmentation/images/d60153c0-db67-11e8-9658-0242ac1c0002.jpg'
img_path16 = 'nails_segmentation/images/a3a73edd-1483-4413-addb-9a7264b5d853.jpg'
img_path17 = 'nails_segmentation/images/d6072ec6-db67-11e8-9658-0242ac1c0002.jpg'
img_path18 = 'nails_segmentation/images/d633f320-db67-11e8-9658-0242ac1c0002.jpg'

def equalizeHistogramRGB(image):
    R, G, B = cv2.split(img)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    return cv2.merge((output1_R, output1_G, output1_B))

def extractSkin(image):

    img = image.copy()

    # equ = equalizeHistogramRGB(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # get HSV thresholds for skin tone in HSV
    lower = np.array([0, 28, 80], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)

    # img[:,:,2] = 180

    # single channel mask, detect skin on the range of lower and upper pixel values in the HSV colorspace.
    mask = cv2.inRange(img, lower, upper)

    # bluring image to improve masking
    # mask = cv2.GaussianBlur(mask, (3, 3), 0)
    cv2.imshow("mask", mask)
    cv2.waitKey(0); cv2.destroyAllWindows()
    kclose = np.ones((3,3), dtype=np.uint8)
    kopen = np.ones((3,3), dtype=np.uint8)


    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kclose, iterations=1)
    cv2.imshow("closing", closing)
    cv2.waitKey(0); cv2.destroyAllWindows()

    # cv2.waitKey(0); cv2.destroyAllWindows()
    # extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=closing)

    # return image of skin
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

# INPUT: image in BGR
# OUTPUT: 3D plot of RGB color space of an image
def plot_rgb_3d(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.show()

    cv2.waitKey(0); cv2.destroyAllWindows()

    r, g, b = cv2.split(rgb_image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = rgb_image.reshape((np.shape(rgb_image)[0]*np.shape(rgb_image)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()

# INPUT: image in BGR
# OUTPUT: 3D plot of HSV color space of an image
def plot_hsv_3d(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    plt.imshow(hsv_image)
    plt.show()

    h, s, v = cv2.split(hsv_image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()

# required for hsvTracker()
def nothing(x):
    pass

# INPUT: image in BGR
# OUTPUT: window with tracking bars that allow to play with HSV channels
def hsvTracker(path):

    # Create a window
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
    cv2.createTrackbar('SMin','image',0,255,nothing)
    cv2.createTrackbar('VMin','image',0,255,nothing)
    cv2.createTrackbar('HMax','image',0,179,nothing)
    cv2.createTrackbar('SMax','image',0,255,nothing)
    cv2.createTrackbar('VMax','image',0,255,nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 20)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    cv2.setTrackbarPos('HMin', 'image', 0)
    cv2.setTrackbarPos('SMin', 'image', 48)
    cv2.setTrackbarPos('VMin', 'image', 80)
    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    img = cv2.imread(path)
    output = img
    waitTime = 33

    while(1):

        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')

        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(img,img, mask= mask)

        # Print if there is a change in HSV value
        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow('image',output)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
def flood_fill(image):
    # set values equal to or above 220 to 0
    # set values below 220 to 255
    threshold, image_threshold = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV);

    # copy the thresholded image
    image_floodfill = image_threshold.copy()

    # mask used to flood filling
    # notice the size needs to be 2 pixels than the image
    h, w = image_threshold.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # floodfill from point (0, 0) - actually black border
    cv2.floodFill(image_floodfill, mask, (0,0), 255);

    # invert floodfilled image
    image_floodfill_inv = cv2.bitwise_not(image_floodfill)

    # combine the two images to get the foreground
    image_out = image_threshold | image_floodfill_inv

    # return images
    return np.hstack((image_threshold, image_floodfill, image_floodfill_inv, image_out))


# https://github.com/CHEREF-Mehdi/SkinDetection/blob/master/SkinDetection.py
def extractSkin3(image):

    img = image.copy()

    # required, otherwise contours and flood fill stop too early
    img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # converting from BGR to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # skin color range for HSV color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)

    # get contours for image
    contours, hierarchy = cv2.findContours(YCrCb_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get the biggest contour
    # cont_max = max(contours, key=cv2.contourArea)
    cont_top = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # cv2.drawContours(img, cont_max, -1, (0,255,0), 3)

    # return result
    return YCrCb_result

    # cv2.imshow("contour", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def bilateral_filtering(image):
    # converting the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smoothing without removing edges
    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)

    # applying the canny filter
    edges = cv2.Canny(gray, 60, 120)
    edges_filtered = cv2.Canny(gray_filtered, 60, 120)

    # stacking the images to print them together for comparison
    images = np.hstack((gray, edges, edges_filtered))

    # return resulting frame
    return images

# main loop
for image_name in images_names:
    img = cv2.imread(images_dir + image_name)

    cv2.imshow("img", img)
    cv2.waitKey(0); cv2.destroyAllWindows()

    skin = extractSkin3(img)
    cv2.imshow("skin", skin)
    cv2.waitKey(0); cv2.destroyAllWindows()

    hand = flood_fill(skin)
    cv2.imshow("floodfill", hand)
    cv2.waitKey(0); cv2.destroyAllWindows()

    # edges = bilateral_filtering(skin)
    # cv2.imshow("edges", edges)
    # cv2.waitKey(0); cv2.destroyAllWindows()

sys.exit()

# modify V values
# rows,cols,pix = img.shape
# for i in range(rows):
#     for j in range(cols):
#         img[i,j] = img[i,j] * (v[i][j] / 255)


# convert from BGR (opencv defatult) to HSV
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# fixed V
# img[:,:,2] = 180
# img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)