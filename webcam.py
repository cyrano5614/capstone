import cv2
import numpy as np

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    # print(frame.shape)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(frame_gray, 50, 150)
    # print(frame_gray.shape)
    # frame_final = np.vstack((frame, frame_gray))
    frame_final = edges
    cv2.imshow("preview", frame_final)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")
