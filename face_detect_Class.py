import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


class FaceRecognition:
    def __init__(self, path):
        self.path = path
        self.images = []
        self.names = []
        self.myList = os.listdir(self.path)
        print(self.myList)
        for cl in self.myList:
            curImg = cv2.imread(f'{self.path}/{cl}')
            self.images.append(curImg)
            self.names.append(os.path.splitext(cl)[0])
        print(self.names)
        self.encodeListKnown = self.find_encodings(self.images)
        print('Encoding Complete')

    def find_encodings(self, images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def mark_presence(self, name):
        with open('presence.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name}, {dtString}')

    def recognize_faces(self):
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture frame from camera. Exiting...")
                break
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encode = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encode, facesCurFrame):
                matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = self.names[matchIndex].upper()
                    print(name)

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), -1)   # cv2.Filled
                    cv2.putText(img, name, (x1+6, y2+6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    self.mark_presence(name)

            cv2.imshow('webcam', img)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()



# face_recog = FaceRecognition(path='LoadImages')
# face_recog.recognize_faces()
