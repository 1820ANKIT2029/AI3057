import os
import cv2
from matplotlib import pyplot as plt
import uuid

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'archor')


if not os.path.exists(POS_PATH):
    os.makedirs(POS_PATH)
if not os.path.exists(NEG_PATH):
    os.makedirs(NEG_PATH)
if not os.path.exists(ANC_PATH):
    os.makedirs(ANC_PATH)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    frame = frame[120:120+250, 200:200+250]

    if cv2.waitKey(1) & 0XFF == ord('a'):
       imgname = os.path.join(ANC_PATH, f'{uuid.uuid1()}.jpg')
       cv2.imwrite(imgname, frame)

       
    if cv2.waitKey(1) & 0XFF == ord('p'):
       imgname = os.path.join(POS_PATH, f'{uuid.uuid1()}.jpg')
       cv2.imwrite(imgname, frame)

    cv2.imshow('Image Collection', frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows() 

plt.imshow(frame)
plt.show()

print(frame.shape)