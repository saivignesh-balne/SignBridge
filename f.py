import cv2

# Try to open the camera
cap = cv2.VideoCapture(0)  # Use index 0 for the default camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully.")

    # Try to capture a single frame
    ret, frame = cap.read()
    if ret:
        print("Frame captured successfully.")
    else:
        print("Error: Could not capture frame.")

# Release the camera
cap.release()
