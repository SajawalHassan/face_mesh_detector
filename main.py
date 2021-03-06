import cv2 as cv
import mediapipe as mp
import time
import face_mesh_module as face_mesh

# Capturing vid (change filename to 0 if need webcam)
capture = cv.VideoCapture(0)
pTime = 0

while True:
    # Reading currunt img
    success, img = capture.read()

    # If can't read currunt img, break loop
    if not success:
          break

    detector = face_mesh.FaceMeshDetector()

    detector.detectFaceMesh(img)

    # Calculating fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    # Displaying fps
    cv.putText(img, f"Fps: {int(fps)}", (10, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv.imshow("Video", img)
    key = cv.waitKey(20)

    if key==27:
        break # If key is pressed, break loop

capture.release()
cv.destroyAllWindows()
