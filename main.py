import cv2 as cv
import mediapipe as mp

# Capturing vid (change filename to 0 if need webcam)
capture = cv.VideoCapture("videos/vid_test_smile.3gp")

while True:
    # Reading currunt img
    success, img = capture.read()

    # If can't read currunt img, break loop
    if not success:
          break


    cv.imshow("Video", img)
    key = cv.waitKey(20)

    if key==27:
        break # If key is pressed, break loop

capture.release()
cv.destroyAllWindows()
