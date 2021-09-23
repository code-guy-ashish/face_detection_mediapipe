import cv2
import mediapipe as mp
import time

ptime = 0

cam = cv2.VideoCapture(0)
wcam, hcam = 1280, 720
cam.set(3, wcam)
cam.set(4, hcam)
myfaceMesh = mp.solutions.face_mesh
faceMesh = myfaceMesh.FaceMesh(max_num_faces=2)
mdraw = mp.solutions.drawing_utils
drawspec = mdraw.DrawingSpec((0, 255, 0), 1, 2)
while True:
    img = cam.read()[1]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(rgb_img)

    if results.multi_face_landmarks:
        for each_face in results.multi_face_landmarks:
            mdraw.draw_landmarks(img, each_face, myfaceMesh.FACEMESH_CONTOURS, drawspec, drawspec)
            for id, lm in enumerate(each_face.landmark):  # To print the landmark of each face
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                #cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                #            cv2.FILLED)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)) + ' fps', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.FILLED)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) == ord('q'):
        break
