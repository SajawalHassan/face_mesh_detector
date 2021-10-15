import cv2 as cv
import mediapipe as mp

class FaceMeshDetector():
    def __init__(self, static_img=False, max_faces=1, refine_landmarks=False, detection_con=0.5,
     tracking_con=0.5):
        self.static_img = static_img,
        self.max_faces = max_faces,
        self.refine_landmarks = refine_landmarks,
        self.detection_con = detection_con,
        self.tracking_con = tracking_con

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_img, max_faces, refine_landmarks, 
        detection_con, tracking_con)

        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
    
    def detectFaceMesh(self, img, draw=True):
        # Convert img rgb for mediapipe
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                
        # Feed results to mediapipe
        results = self.faceMesh.process(imgRGB)

        faces = results.multi_face_landmarks

        if draw:
            if faces:
                for face in faces:
                    self.mpDraw.draw_landmarks(img, face, self.mpFaceMesh.FACEMESH_CONTOURS,
                    self.drawSpec, self.drawSpec)

