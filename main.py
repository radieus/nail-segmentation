import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
import random as rng

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt

mypath = 'nails_segmentation/images/'
images = [f for f in listdir(mypath) if isfile(join(mypath, f))]
img_path1 = 'nails_segmentation/images/1eecab90-1a92-43a7-b952-0204384e1fae.jpg'
img_path2 = 'nails_segmentation/images/2C29D473-CCB4-458C-926B-99D0042161E6.jpg'
img_path3 = 'nails_segmentation/images/2c376c66-9823-4874-869e-1e7f5c54ec7b.jpg'
img_path4 = 'nails_segmentation/images/4c4a0dd6-e402-11e8-97db-0242ac1c0002.jpg'
img_path5 = 'nails_segmentation/images/d60a5f06-db67-11e8-9658-0242ac1c0002.jpg'

# no nails
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

def extractSkin(image):

    img = image.copy()
    # convert from BGR (opencv defatult) to HSV
    R, G, B = cv2.split(img)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    equ = cv2.merge((output1_R, output1_G, output1_B))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # get HSV thresholds for skin tone in HSV
    lower = np.array([0, 48, 80], dtype=np.uint8)
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


    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kopen, iterations=3)
    cv2.imshow("opening", opening)
    cv2.waitKey(0); cv2.destroyAllWindows()

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kclose, iterations=6)
    cv2.imshow("closing", closing)
    cv2.waitKey(0); cv2.destroyAllWindows()


    # contours,hierarchy = cv2.findContours(closing, 1, 2)
    # contours_sizes= [(cv2.contourArea(cnt), cnt) for cnt in contours]
    # biggest_contour = max(contours_sizes, key=lambda x: x[0])[1]

    # countours = biggest_contour
    # cv2.imshow("contours", np.array(contours))

    #cv2.waitKey(0); cv2.destroyAllWindows()
    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=closing)

    # return image of skin
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


# INPUT: image in BGR
# OUTPUT: 3D plot of RGB color space
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
# OUTPUT: 3D plot of HSV color space
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

def find_biggest_contour(image):
   image = image.copy() 
   #1
   image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
   #2 
   threshold = cv2.threshold(image_gray,127, 255,0)
   #3
   contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # countours is a python list
   contours_sizes= [(cv2.contourArea(cnt), cnt) for cnt in contours]
   biggest_contour = max(contours_sizes, key=lambda x: x[0])[1]
   #define a mask
   mask = np.zeros(image.shape, np.uint8)
   cv2.drawContours(mask,[biggest_contour], -1, (0,255,0), 3)# 3=thickness, -1= draw all contours, 2nd arg must be a list 
   return biggest_contour, mask

def contour(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    epsilon = 0.1 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    cv2.drawContours(image, approx, -1, (0, 255, 0), 3)
    cv2.imshow("Contour", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extractSkin2(image):
    img = image.copy()

    # convert from BGR (opencv defatult) to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # fixed V
    img[:,:,2] = 180
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            
    cv2.imshow('fixed v', img)
    cv2.waitKey(0); cv2.destroyAllWindows()

    # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
    # Use a bimodal image as an input.
    # Optimal threshold value is determined automatically.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu_threshold, mask = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # bluring image to improve masking
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=mask)


    # img = cv2.equalizeHist(img)
    # cv2.imshow('histo equalization', img)
    # cv2.waitKey(0); cv2.destroyAllWindows()

    # print("Obtained threshold: ", otsu_threshold)
    # cv2.imshow("otsu", otsu_img)
    # cv2.waitKey(0); cv2.destroyAllWindows()

    # return image of skin
    return mask

def thresh_callback(val, image):

    src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = val
    # # Detect edges using Canny
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)
    # Show in a window

    return drawing

def bilateral_filtering(image):
    # Converting the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smoothing without removing edges.
    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)

    # Applying the canny filter
    edges = cv2.Canny(gray, 60, 120)
    edges_filtered = cv2.Canny(gray_filtered, 60, 120)

    # Stacking the images to print them together for comparison
    images = np.hstack((gray, edges, edges_filtered))

    # Display the resulting frame
    cv2.imshow('Frame', images)

# hsvTracker(img_path13)

for image in images:
    img = cv2.imread(mypath + image)
    skin = extractSkin(img)
    bilateral_filtering(skin)
    # skin = extractSkin(img)
    # cv2.imshow("skin", skin)
    cv2.waitKey(0); cv2.destroyAllWindows()
    # contour(skin)

    # cv2.imshow("thresh", thresh_callback(100, skin))
    # cv2.waitKey(0); cv2.destroyAllWindows()


sys.exit()

# modify V values
# rows,cols,pix = img.shape
# for i in range(rows):
#     for j in range(cols):
#         img[i,j] = img[i,j] * (v[i][j] / 255)