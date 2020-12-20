import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join

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

def extractSkin(image):

    img = image.copy()
    # convert from BGR (opencv defatult) to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # get HSV thresholds for skin tone in HSV
    lower = np.array([0, 48, 80], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)

    # single channel mask, detect skin on the range of lower and upper pixel values in the HSV colorspace.
    mask = cv2.inRange(img, lower, upper)

    # bluring image to improve masking
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=mask)

    # contours, hierarychy = cv2.findContours

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


def extractSkin2(image):
    img = image.copy()
    # convert from BGR (opencv defatult) to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    v = img[:,:,2]

    # fixed V (?)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    rows,cols,pix = img.shape
    for i in range(rows):
        for j in range(cols):
            img[i,j] = img[i,j] * (v[i][j] / 255)
            
    # cv2.imshow('fixed v', img)
    # cv2.waitKey(0); cv2.destroyAllWindows()

    # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
    # Use a bimodal image as an input.
    # Optimal threshold value is determined automatically.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu_threshold, otsu_img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # print("Obtained threshold: ", otsu_threshold)
    # cv2.imshow("otsu", otsu_img)
    # cv2.waitKey(0); cv2.destroyAllWindows()

    # return image of skin
    return otsu_img


for image in images:
    img = cv2.imread(mypath + image)
    cv2.imshow("img", extractSkin(img))
    cv2.waitKey(0); cv2.destroyAllWindows()

sys.exit()