import cv2

detectorFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

videoCapture = cv2.VideoCapture(0)

while True:
    ok, frame = videoCapture.read()

    imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    deteccoes = detectorFace.detectMultiScale(imagemCinza, minSize=(80, 80))

    for (x, y, w, h) in deteccoes:
        print(w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()







