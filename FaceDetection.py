import cv2
import mediapipe as mp
import time

ptime = 0
#####################
wcam, hcam = 1280, 720
####################


cam = cv2.VideoCapture(0)  #To use the camera feed
cam.set(3, wcam)
cam.set(4, hcam)

myface = mp.solutions.face_detection
face = myface.FaceDetection(model_selection=1)
mdraw = mp.solutions.drawing_utils

while True:
    img = cam.read()[1]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face.process(rgb_img)

    if results.detections:
        for id, det in enumerate(results.detections):
            #mdraw.draw_detection(img,det)
            h, w, c = img.shape
            bboxc = det.location_data.relative_bounding_box
            bbox = int(bboxc.xmin * w), int(bboxc.ymin * h), int(bboxc.width * w), int(bboxc.height * h)
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, f'{(int(det.score[0]*100))}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.FILLED)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps))+' fps', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.FILLED)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) == ord('q'):
        break
