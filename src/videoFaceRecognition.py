import cv2
from deepface import DeepFace

if __name__ == '__main__':

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open Webcam")

    while True:
        ret, frame = cap.read()

        result = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        xBox=0
        yBox=0
        emocion=""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            xBox=x
            yBox=y
        font = cv2.FONT_HERSHEY_DUPLEX
        emocion = result['dominant_emotion']
        cv2.putText(frame, emocion, (xBox, yBox - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1, 1)
        cv2.imshow('Original Video', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
