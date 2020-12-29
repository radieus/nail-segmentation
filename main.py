import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import isfile, join
from sklearn.metrics import jaccard_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

images_dir = 'nails_segmentation/images/'
labels_dir = 'nails_segmentation/labels/'
images_names = [f for f in listdir(images_dir) if isfile(join(images_dir, f))]

# some images
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


# showing images
def show(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0); cv2.destroyAllWindows()

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

    # invert the image mask
    threshold, image_threshold = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV);

    # operate on a copy of image mask
    image_floodfill = image_threshold.copy()

    # mask used to flood-filling
    h, w = image_threshold.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # flood-fill from seed (0, 0) - actually black border
    cv2.floodFill(image_floodfill, mask, (0,0), 255);

    # invert flood-filled image
    image_floodfill_inv = cv2.bitwise_not(image_floodfill)

    # combine the two images to get the foreground
    image_out = image_threshold | image_floodfill_inv

    # show("floodfill", np.hstack((image_threshold, image_floodfill, image_floodfill_inv, image_out)))

    # delete previously created border
    h, w = image_out.shape[:2]
    borderless_img = image_out[2:h-2, 2:w-2]

    # return mask
    return borderless_img


# https://github.com/CHEREF-Mehdi/SkinDetection/blob/master/SkinDetection.py
def extractSkin(image):

    # operate on a copy
    img = image.copy()

    # required, otherwise contours/flood-fill will stop too early
    img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # converting from BGR to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    
    # skin color range for YCrCb color space 
    lower = (0, 135, 85)
    upper = (255, 180, 135)
    YCrCb_mask = cv2.inRange(img_YCrCb, lower, upper) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)

    # return mask
    return YCrCb_result

def iou_score(label, image):

    # operate on a copy
    label = label.copy()

    # convert label to grayscale
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

    #calculate IoU
    intersection = np.logical_and(label, image)
    union = np.logical_or(label, image)
    iou_score = np.sum(intersection) / np.sum(union)

    # return score
    return iou_score

def test():
    avg = 0
    cnt = 0
    n = 0
    for image_name in images_names:
        n += 1
        label = cv2.imread(labels_dir + image_name)
        img = cv2.imread(images_dir + image_name)
        hand_mask = flood_fill(extractSkin(img))
        hand = cv2.bitwise_and(img, img, mask=hand_mask)
        src = cv2.medianBlur(hand, 21)
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

        # purplish
        lower = np.array([80, 0, 113])
        upper = np.array([180, 105, 255])
        mask1 = cv2.inRange(hsv, lower, upper)

        # orangish
        lower = np.array([0, 0, 180])
        upper = np.array([7, 94, 255])
        mask2 = cv2.inRange(hsv, lower, upper)

        mask = mask1 | mask2

        kernel_open = np.ones((4,4),np.uint8)
        kernel_close = np.ones((7,7),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        iou = iou_score(label, mask)
        print(f"{image_name}: {iou}")
        cnt += iou
    avg = cnt / n
    print(f"average: {avg}")

# main loop
def main():
    for image_name in images_names:
        label = cv2.imread(labels_dir + image_name)
        img = cv2.imread(images_dir + image_name)
        show("img", img)

        # get skin
        skin = extractSkin(img)
        show("skin", (255-skin))

        # get hand mask
        hand_mask = flood_fill(skin)
        show("hand_mask1", hand_mask)

        # use hand mask on the original image to get hand
        hand = cv2.bitwise_and(img, img, mask=hand_mask)
        show("hand", hand)

        # blur hand
        hand = cv2.medianBlur(hand, 21)
        show("hand", hand)

        # convert to HSV
        hsv = cv2.cvtColor(hand, cv2.COLOR_BGR2HSV)
        show("hsv", hsv)

        # get pinkish color
        lower = np.array([70, 0, 131])
        upper = np.array([180, 105, 255])
        mask1 = cv2.inRange(hsv, lower, upper)

        lower = np.array([0, 0, 180])
        upper = np.array([7, 94, 255])
        mask2 = cv2.inRange(hsv, lower, upper)

        mask = mask1 | mask2
        show("mask", mask)

        # get image properties
        # height, width, channels = hand.shape

        # create blob detector params object and set its parameters (we want circular/convex object)
        # params = cv2.SimpleBlobDetector_Params()
        # params.filterByArea = True
        # params.minArea = int((height * width) / 500)
        # params.maxArea = int((height * width) / 10)
        # params.filterByCircularity = True
        # params.minCircularity = 0.5
        # params.filterByConvexity = True
        # params.minConvexity = 0.5
        # params.filterByInertia = True
        # params.minInertiaRatio = 0.05

        # # create blob detector object and pass params
        # detector = cv2.SimpleBlobDetector_create(params)
        # key_points = detector.detect(255 - mask)

        # vis = cv2.bitwise_and(hsv, hsv, mask=mask)
        # cv2.imshow("vis", vis)
        # cv2.waitKey(0); cv2.destroyAllWindows()

        # vis = cv2.addWeighted(src, 0.2, vis, 0.8, 0)
        # cv2.imshow("vis", vis)
        # cv2.waitKey(0); cv2.destroyAllWindows()

        # # drawing keypoints
        # cv2.drawKeypoints(vis, key_points, vis, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # for kp in key_points:
        #     cv2.drawMarker(vis, (int(kp.pt[0]), int(kp.pt[1])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=3)

        # cv2.imshow("VIS", vis)
        # cv2.waitKey(0); cv2.destroyAllWindows()

        print(f"{image_name}: {iou_score(label, mask)}")

test()
# main()
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

# get contours for image
# contours, hierarchy = cv2.findContours(YCrCb_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# get the biggest contour
# cont_max = max(contours, key=cv2.contourArea)
# cont_top = sorted(contours, key=cv2.contourArea, reverse=True)[0]
# cv2.drawContours(img, cont_max, -1, (0,255,0), 3)