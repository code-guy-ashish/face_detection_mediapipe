import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selction = model_selection
        self.myface = mp.solutions.face_detection
        self.face = self.myface.FaceDetection(self.min_detection_confidence, self.model_selction)
        self.mdraw = mp.solutions.drawing_utils

    def findFace(self, img, draw=True):

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.face.process(rgb_img)
        bboxs = []
        if self.results.detections:
            for id, det in enumerate(self.results.detections):
                # mdraw.draw_detection(img,det)
                h, w, c = img.shape
                bboxc = det.location_data.relative_bounding_box
                bbox = int(bboxc.xmin * w), int(bboxc.ymin * h), int(bboxc.width * w), int(bboxc.height * h)
                bboxs.append([id, bbox, det.score])
                cv2.rectangle(img, bbox, (0, 255, 0), 2)
                cv2.putText(img, f'{(int(det.score[0] * 100))}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2, cv2.FILLED)
        return img, bboxs


def main():
    ptime = 0
    cam = cv2.VideoCapture(0)
    wcam, hcam = 1280, 720
    cam.set(3, wcam)
    cam.set(4, hcam)
    detector = FaceDetector()
    while True:
        img = cam.read()[1]
        img, bboxes = detector.findFace(img)
        print(bboxes)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)) + ' fps', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.FILLED)
        cv2.imshow('Image', img)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()
