import cv2
import numpy as np

def detect_circles(image, min_radius, max_radius):
    # Konversi gambar ke grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Lakukan deteksi tepi menggunakan Canny
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Lakukan transformasi Hough untuk mendeteksi lingkaran
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        # Ubah koordinat lingkaran menjadi format integer
        circles = np.uint16(np.around(circles))

        # Gambar lingkaran yang terdeteksi
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv2.circle(image, center, i[2], (0, 255, 0), 2)

    return image

def main():
    # Baca gambar
    image = cv2.imread('new.jpg')
    cv2.imshow('Original Image', image)

    # Tentukan rentang radius lingkaran yang akan dideteksi
    min_radius = 10
    max_radius = 100

    # Deteksi lingkaran menggunakan metode Hough
    result_image = detect_circles(image, min_radius, max_radius)

    # Tampilkan hasil    
    cv2.imshow('Detected Circles', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
