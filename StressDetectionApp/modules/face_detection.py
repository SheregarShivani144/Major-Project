import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("models/face_model.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face_stress():

    cap = cv2.VideoCapture(0)

    stress_score = 0

    while True:

        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (48, 48))

            face = face / 255.0

            face = np.reshape(face, (1, 48, 48, 1))

            prediction = model.predict(face)

            emotion = np.argmax(prediction)

            if emotion in [3,4,5]:
                stress_score = 2
            else:
                stress_score = 0

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()

    cv2.destroyAllWindows()

    return stress_score