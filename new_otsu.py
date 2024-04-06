import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

# Tesseract OCR 실행 파일 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 이미지의 채도를 높이는 함수
def increase_saturation_hsv(img, saturation_increase=50):
    # 이미지를 HSV 색 공간으로 변환
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # S(채도) 채널에서 채도를 증가시키기
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1].astype(int) + saturation_increase, 0, 255)
    
    # 변환된 이미지를 다시 BGR 색 공간으로 변환하여 반환
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)


# 이미지 로드
img = cv2.imread(r'data/c6.png')

# flag 설정
apply_saturation_increase = False

if apply_saturation_increase:
    # 채도를 높이기 위한 상수 설정
    saturation_increase = 255

    # 이미지의 채도 높이기
    img = increase_saturation_hsv(img, saturation_increase)

# 이미지를 그레이스케일로 변환
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이미지에 Otsu의 임계값 적용
_, img_otsu_first = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Otsu의 임계값을 기준으로 검정 부분과 흰색 부분 분리
black_part = np.where(img_otsu_first == 0)
white_part = np.where(img_otsu_first == 255)

# 검정 부분을 제외하고 흰색 부분에 대해 다시 Otsu의 임계값 적용
img_white = gray_img.copy()
img_white[black_part] = 255
_, img_otsu_second = cv2.threshold(img_white, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 검정 부분과 두 번째 Otsu의 임계값을 적용한 부분을 합침
img_combined = img_otsu_first.copy()
img_combined[white_part] = img_otsu_second[white_part]


# 그래프 그리기
plt.figure(figsize=(18, 6))

# 원본 이미지
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# 밝힌 이미지
plt.subplot(1, 4, 2)
plt.imshow(gray_img, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# 첫 번째 Otsu 적용 이미지
plt.subplot(1, 4, 3)
plt.imshow(img_otsu_first, cmap='gray')
plt.title('First Otsu Thresholded Image')
plt.axis('off')

# 두 번째 Otsu 적용 이미지
plt.subplot(1, 4, 4)
plt.imshow(img_combined, cmap='gray')
plt.title('Second Otsu Thresholded Image (excluding black areas)')
plt.axis('off')

plt.show()


text_first = pytesseract.image_to_string(img_otsu_first, config='--oem 3 --psm 6 -l kor')
text_first = text_first.replace(" ", "")
text_first = text_first.replace("\n", "")
print("OCR 결과 (첫 번째 이진화 후):", text_first)

# text_second = pytesseract.image_to_string(img_otsu_second, config='--oem 3 --psm 6 -l kor')
# text_second = text_second.replace(" ", "")
# text_second = text_second.replace("\n", "")
# print("OCR 결과 (두 번째 이진화 후):", text_second)

text_second = pytesseract.image_to_string(img_combined, config='--oem 3 --psm 6 -l kor')
text_second = text_second.replace(" ", "")
text_second = text_second.replace("\n", "")
print("OCR 결과 (두 번째 이진화 후):", text_second)

