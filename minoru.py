import numpy as np
import cv2

left_camera = cv2.VideoCapture(4)
right_camera = cv2.VideoCapture(6)

# id cam l = 4
# id cam r = 6

# Check if the cameras opened successfully
if not left_camera.isOpened() or not right_camera.isOpened():
    print("Error: Unable to open one or both cameras")
    exit()

# Buat objek StereoSGBM
stereo = cv2.StereoSGBM_create(numDisparities=3, blockSize=15)

while True:
     # Ambil frame dari kedua kamera
    ret1, left_frame = left_camera.read()
    ret2, right_frame = right_camera.read()

    if not (ret1 and ret2):
        break
    # Konversi frame menjadi citra abu-abu
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    try:
     # Hitung disparitas
        disparity = stereo.compute(left_gray, right_gray) + 0.0000001
    except cv2.error as e:
        print(f"Error computing disparity: {e}")
        continue

    # Hitung jarak ke objek
    # Faktor konversi yang bergantung pada konfigurasi kamera 
    depth = 1.0 / disparity

     # Tampilkan citra kedua kamera dan citra kedalaman
    cv2.imshow('Camera Laptop', left_frame)
    cv2.imshow('Camera HP', right_frame)
    cv2.imshow('Depth Map', depth)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left_camera.release()
right_camera.release()
cv2.destroyAllWindows()