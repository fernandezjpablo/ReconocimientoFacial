import cv2
from deepface import DeepFace

if __name__ == '__main__':

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture("videos/emociones.mp4")

    while(cap.isOpened()):
        ret, frame = cap.read()

        result = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        xBox = 0
        yBox = 0
        emocionAngry = 0
        emocionDisgust = 0
        emocionFear = 0
        emocionHappy = 0
        emocionSad = 0
        emocionSurprise = 0
        emocionNeutral = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            xBox = x
            yBox = y
        font = cv2.FONT_HERSHEY_DUPLEX
        if result['dominant_emotion'] == "Angry":
            emocionAngry = emocionAngry + 1
        elif result['dominant_emotion'] == "Disgust":
            emocionDisgust = emocionDisgust + 1
        elif result['dominant_emotion'] == "Fear":
            emocionFear = emocionFear + 1
        elif result['dominant_emotion'] == "Happy":
            emocionHappy = emocionHappy + 1
        elif result['dominant_emotion'] == "Sad":
            emocionSad = emocionSad + 1
        elif result['dominant_emotion'] == "Surprise":
            emocionSurprise = emocionSurprise + 1
        elif result['dominant_emotion'] == "Neutral":
            emocionNeutral = emocionNeutral + 1

        age = result['age']
        gender = result['gender']
        race = result['race']
        cv2.putText(frame, result['dominant_emotion'], (xBox, yBox - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1,
                    1)
        cv2.imshow('Original Video', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(emocionHappy, " , ", emocionSad, " , ", emocionFear, " , ", emocionAngry, " , ", age, " , ", gender, " , ",
          race)
