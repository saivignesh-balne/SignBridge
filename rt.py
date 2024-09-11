import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import math

# Load the pre-trained model
model = tf.keras.models.load_model('Data/model.h5')  # Change to 'my_model.keras' if needed

# Define class labels according to your model's classes
class_labels = ['hi', 'iloveu', 'yes']

# Initialize the webcam and hand detector
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands=1)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Detect hands in the frame
    hands, img = detector.findHands(frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a blank background
        bg = np.ones((300, 300, 3), np.uint8) * 255
        imgCrop = frame[y-20:y + h+20, x-20:x + w+20]

        # Resize based on aspect ratio
        aspectRatio = h / w
        if aspectRatio > 1:
            k = 300 / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, 300))
            wGap = math.ceil((300 - wCal) / 2)
            bg[:, wGap:wCal + wGap] = imgResize
        else:
            k = 300 / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (300, hCal))
            hGap = math.ceil((300 - hCal) / 2)
            bg[hGap:hCal + hGap, :] = imgResize

        # Preprocess the image for the model
        img_preprocessed = cv2.resize(bg, (256, 256))
        img_preprocessed = img_preprocessed.astype(np.float32) / 255.0
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

        # Make predictions
        yhat = model.predict(img_preprocessed)
        predicted_class = np.argmax(yhat, axis=1)
        label = class_labels[predicted_class[0]]

        # Display the result on the frame
        cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the cropped hand and background
        #cv2.imshow("Hands", imgCrop)
        #cv2.imshow("BG", bg)

    # Show the main camera feed
    cv2.imshow("Camera", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
