import cv2
import numpy as np

def generate_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def generate_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # Tambahan level terakhir dari Gaussian Pyramid
    return laplacian_pyramid

def main():
    # Baca gambar
    image = cv2.imread('img.jpg')

    # Konversi gambar ke grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur gambar menggunakan filter Gaussian
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Tentukan jumlah level pyramid
    levels = 4

    # Generate Gaussian Pyramid
    gaussian_pyramid = generate_gaussian_pyramid(blurred_image, levels)

    # Generate Laplacian Pyramid
    laplacian_pyramid = generate_laplacian_pyramid(gaussian_pyramid)

    # Tampilkan hasil
    for i in range(levels):
        cv2.imshow(f'Gaussian Pyramid Level {i}', gaussian_pyramid[i])
        cv2.imshow(f'Laplacian Pyramid Level {i}', laplacian_pyramid[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
