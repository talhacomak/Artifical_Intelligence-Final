import imutils
import numpy as np

#
# Computes mean square error between two n-d matrices. Lower = more similar.
#
def meanSquareError(img1, img2):
    assert img1.shape == img2.shape, "Images must be the same shape."
    error = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    error = error/float(img1.shape[0] * img1.shape[1] * img1.shape[2])
    return error

def compareImages(img1, img2):
    return 1/meanSquareError(img1, img2)


#
# Computes pyramids of images (starts with the original and down samples).
# Adapted from:
# http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
#
def pyramid(image, scale = 1.5, minSize = 30, maxSize = 1000):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width = w)
        if(image.shape[0] < minSize or image.shape[1] < minSize):
            break
        if (image.shape[0] > maxSize or image.shape[1] > maxSize):
            continue
        yield image

#
# "Slides" a window over the image. See for this url for cool animation:
# http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
#
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y+windowSize[1], x:x+windowSize[1]])

