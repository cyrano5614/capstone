import cv2
# import numpy as np
from skimage.feature import hog

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    # print(frame.shape)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(frame_gray, 50, 150)
    frame_gray = cv2.resize(frame_gray, (128, 128))
    hog_array, hog_image = hog(frame_gray, visualise=True,
                               block_norm='L2-Hys')
    # print(frame_gray.shape)
    # frame_final = np.vstack((frame, frame_gray))
    frame_final = edges
    cv2.imshow("preview", hog_image)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
cv2.destroyWindow("preview")
