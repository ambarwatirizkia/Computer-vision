import cv2
import numpy as np

def calculate_ssd_with_filtering(image, template, filter_kernel):
    # Ubah gambar dan template menjadi grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_image', gray_image)   
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Terapkan filtering h[m, n]
    filtered_image = cv2.filter2D(gray_image, -1, filter_kernel)
    cv2.imshow('filter', filtered_image)

    # Metode Template Matching dengan Sum of Squared Differences (SSD)
    result_ssd = cv2.matchTemplate(filtered_image, gray_template, cv2.TM_SQDIFF_NORMED)
    min_val_ssd, _, min_loc_ssd, _ = cv2.minMaxLoc(result_ssd)

    # Gambar kotak di sekitar hasil pencocokan SSD
    top_left_ssd = min_loc_ssd
    bottom_right_ssd = (top_left_ssd[0] + template.shape[1], top_left_ssd[1] + template.shape[0])
    cv2.rectangle(image, top_left_ssd, bottom_right_ssd, 255, 2)

    # Tampilkan gambar asli dengan kotak pencocokan setelah filtering
    cv2.imshow('Template Matching (SSD with Filtering)', image)
    
    result_ssd = cv2.matchTemplate(filtered_image, gray_template, cv2.TM_SQDIFF_NORMED)
    min_val_sad, _, _, _ = cv2.minMaxLoc(result_ssd)

    return min_val_sad

# Baca gambar asli dan gambar template
image = cv2.imread('img.jpg')
template = cv2.imread('tmp.jpg')

# Definisikan kernel filter h[m, n]
filter_kernel = np.array([[1, 1, 1],
                          [1, -7, 1],
                          [1, 1, 1]])

# Panggil fungsi calculate_ssd_with_filtering
ssd_value_with_filtering = calculate_ssd_with_filtering(image, template, filter_kernel)

# Tampilkan nilai SAD setelah filtering
print(f"Nilai SSD setelah filtering: {ssd_value_with_filtering}")

cv2.waitKey(0)
cv2.destroyAllWindows()