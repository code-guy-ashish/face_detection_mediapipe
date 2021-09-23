import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.myfaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.myfaceMesh.FaceMesh(self.static_image_mode, self.max_num_faces,
                                                 self.min_detection_confidence, self.min_tracking_confidence)
        self.mdraw = mp.solutions.drawing_utils
        self.drawspec = self.mdraw.DrawingSpec((0, 255, 0), 1, 2)

    def findFaceMesh(self, img, draw=True):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(rgb_img)
        faces = []
        if results.multi_face_landmarks:
            for each_face in results.multi_face_landmarks:
                if draw:
                    self.mdraw.draw_landmarks(img, each_face, self.myfaceMesh.FACEMESH_CONTOURS, self.drawspec,
                                              self.drawspec)
                for id, lm in enumerate(each_face.landmark):  # To print the landmark of each face
                    face = []
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    # cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    #            cv2.FILLED)
                    face.append([cx, cy])
                faces.append(face)

        return img, faces


def main():
    ptime = 0
    cam = cv2.VideoCapture(0)
    wcam, hcam = 1280, 720
    cam.set(3, wcam)
    cam.set(4, hcam)
    detector = FaceMeshDetector()
    while True:
        img = cam.read()[1]
        img, f_aces = detector.findFaceMesh(img)
        if len(f_aces) != 0:
            # print(len(f_aces)) #Can perform any desired operation with faces data.
            pass
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)) + ' fps', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.FILLED)
        cv2.imshow('Image', img)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()
