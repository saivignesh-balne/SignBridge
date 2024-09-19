import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open webcam")
else:
    print("Webcam is open")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
