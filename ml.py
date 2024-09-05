import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow as tf

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
detector = HandDetector(maxHands=1)

model = tf.keras.models.load_model('sih24/Data/keras_model.h5')
classifier = Classifier(model,'sih24/Data/labels.txt')

folder = 'sih24/images/yes'
count = 0
labels = ["Hello","I Love You","Yes"]

while cap.isOpened():
    ret , img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        
        bg = np.ones((300,300,3), np.uint8)*255
        imgCrop = img[y-20:y + h+20, x-20:x + w+20]
        
        imgCropShape = imgCrop.shape
        
        aspectRatio = h/w
        
        if aspectRatio > 1:
            k = 300/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,300))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300-wCal)/2)
            bg[:,wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(img)
            print(prediction,index)
        else:
            k = 300/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(300,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((300-hCal)/2)
            bg[hGap:hCal+hGap,:] = imgResize
        
        cv2.imshow("Hands",imgCrop)
        cv2.imshow("BG",bg)
    
    cv2.imshow("Camera",img)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()