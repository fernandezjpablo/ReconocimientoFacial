import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    img = cv2.imread("H:\Personal\ProyectosPersonales\ReconocimientoFacial\imagenes\happyman.png")
    plt.imshow(img)  # BGR
    plt.show()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    # model = load_model('models/facial_expression_model_weights.h5')
    result = DeepFace.analyze(img, enforce_detection=False)
    print(result)
    print(result['dominant_emotion'])
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, result['dominant_emotion'], (50, 50), font, 3, (0, 255, 0), 2, cv2.LINE_4)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    img = cv2.imread("H:\Personal\ProyectosPersonales\ReconocimientoFacial\imagenes\sadWoman.jpg")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    result = DeepFace.analyze(img, enforce_detection=False)
    print(result)
    print(result['dominant_emotion'])