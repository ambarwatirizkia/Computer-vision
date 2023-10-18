import cv2
import numpy as np

# Open the left and right cameras (modify indices if needed)
left_cap = cv2.VideoCapture(4)
right_cap = cv2.VideoCapture(6)

# Check if the cameras opened successfully
if not left_cap.isOpened() or not right_cap.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Set up stereo block matching parameters (adjust as needed)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)

while True:
    # Read frames from both cameras
    ret1, left_frame = left_cap.read()
    ret2, right_frame = right_cap.read()

    # Check if both frames were read successfully
    if not ret1 or not ret2:
        print("Error: Could not read frames from one or both cameras.")
        break

    # Convert frames to grayscale
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Compute disparity map
    disparity = stereo.compute(left_gray, right_gray)

    # Normalize disparity values for better visualization
    disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity = disparity.astype(np.uint8)

    # Display the disparity map
    cv2.imshow('Disparity Map', disparity)
    cv2.imshow('left frame', left_frame)
    cv2.imshow('right frame', right_frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release both cameras and close the window
left_cap.release()
right_cap.release()
cv2.destroyAllWindows()
