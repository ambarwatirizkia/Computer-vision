import cv2
import numpy as np

def calculate_sad_with_filtering(image, template, filter_kernel):
    # Ubah gambar dan template menjadi grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_image', gray_image)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Terapkan filtering h[m, n]
    filtered_image = cv2.filter2D(gray_image, -1, filter_kernel)
    cv2.imshow('filter', filtered_image)

    # Metode Template Matching dengan Sum of Absolute Differences (SAD)
    result_sad = cv2.matchTemplate(filtered_image, gray_template, cv2.TM_SQDIFF)
    min_val_sad, _, min_loc_sad, _ = cv2.minMaxLoc(result_sad)

    # Gambar kotak di sekitar hasil pencocokan SAD
    top_left_sad = min_loc_sad
    bottom_right_sad = (top_left_sad[0] + template.shape[1], top_left_sad[1] + template.shape[0])
    cv2.rectangle(image, top_left_sad, bottom_right_sad, 255, 2)

    # Tampilkan gambar asli dengan kotak pencocokan setelah filtering
    cv2.imshow('Template Matching (SAD with Filtering)', image)
    
    result_sad = cv2.matchTemplate(filtered_image, gray_template, cv2.TM_SQDIFF)
    min_val_sad, _, _, _ = cv2.minMaxLoc(result_sad)

    return min_val_sad
# Baca gambar asli dan gambar template
image = cv2.imread('img.jpg')
template = cv2.imread('tmp.jpg')

# Definisikan kernel filter h[m, n]
filter_kernel = np.array([[1, 1, 1],
                          [1, -7, 1],
                          [1, 1, 1]])

# Panggil fungsi calculate_sad_with_filtering
sad_value_with_filtering = calculate_sad_with_filtering(image, template, filter_kernel)

# Tampilkan nilai SAD setelah filtering
print(f"Nilai SAD setelah filtering: {sad_value_with_filtering}")

cv2.waitKey(0)
cv2.destroyAllWindows()