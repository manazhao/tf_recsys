import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(
    '/usr/local/share/OpenCV/haarcascades/haarcascade_fullbody.xml')
img = cv2.imread('testdata/player2_small.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minSize=(30, 30), minNeighbors=4)
print faces
for (x, y, w, h) in faces:
  cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
  roi_gray = gray[y:y + h, x:x + w]
  roi_color = img[y:y + h, x:x + w]
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
