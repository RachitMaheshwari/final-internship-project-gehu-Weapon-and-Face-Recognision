import cv2
import numpy as np
from face_detect_Class import FaceRecognition

class WeaponDetector:
    def __init__(self, weights_path, config_path, classes=["laptop", "knife", "handgun", "cigarette"]):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.classes = classes

        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.cap = None


    def start(self, video_path=None):
        if video_path is None:
            video_path = 0
        self.cap = cv2.VideoCapture(video_path)

        while True:
            _, img = self.cap.read()
            height, width, channels = img.shape


            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            if indexes in (0,1,2,3):
                FaceRecognition.face_recog = FaceRecognition(path='LoadImages')
                FaceRecognition.face_recog.recognize_faces()

            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    color = self.colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)



            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()



detector = WeaponDetector("yolov4-custom_last.weights", "yolov4-custom.cfg")
detector.start()
