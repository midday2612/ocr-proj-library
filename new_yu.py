import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

# Tesseract OCR 실행 파일 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 이미지를 YUV 색 공간에서 U 채널을 기준으로 이진화하는 함수 정의
def threshold_yuv_u(img):
    # 이미지를 YUV 색 공간으로 변환
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # U 채널 추출
    u_channel = yuv_img[:, :, 1]
    
    return u_channel

def convert_to_grayscale_yuv(image):
    # 이미지를 읽어옴
    # image = cv2.imread(image_path)

    # 이미지를 YUV로 변환
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # U값만 가져와서 흑백 이미지로 변환
    y_channel = yuv_image[:,:,0]

    return y_channel

# 이미지 로드
img = cv2.imread(r'data/ttest.png')

# 이미지를 BGR에서 YUV로 변환
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

gray_img = convert_to_grayscale_yuv(img)
u_img = threshold_yuv_u(img)

# 이미지에 Otsu의 임계값 적용
_, img_otsu_first = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("Binary Image (Otsu's Thresholding with U channel)", img_otsu_first)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Otsu의 임계값을 기준으로 검정 부분과 흰색 부분 분리
black_part = np.where(img_otsu_first == 0)
white_part = np.where(img_otsu_first == 255)



##################################################################################3
# # # 검정 부분을 제외하고 흰색 부분에 대해 YUV U 채널 기준으로 오츠의 임계값 적용
img_white = gray_img.copy()
img_white[black_part] = 255
cv2.imshow("Binary Image (Otsu's Thresholding with U channel)", img_white)
cv2.waitKey(0)
cv2.destroyAllWindows()
_, img_otsu_second = cv2.threshold(img_white, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("Binary Image (Otsu's Thresholding with U channel)", img_otsu_second)
cv2.waitKey(0)
cv2.destroyAllWindows()

text_second = pytesseract.image_to_string(img_otsu_second, config='--oem 3 --psm 6 -l kor')
text_second = text_second.replace(" ", "")
text_second = text_second.replace("\n", "")
print("OCR 결과 (두 번째 이진화 후):", text_second)

############################################################################
# 흰색 부분을 제외하고 검정 부분에 대해 오츠의 임계값 적용
img_black = gray_img.copy()
img_black[white_part] = 0
cv2.imshow("Binary Image (Otsu's Thresholding with U channel)", img_black)
cv2.waitKey(0)
cv2.destroyAllWindows()
_, img_otsu_second = cv2.threshold(img_black, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("Binary Image (Otsu's Thresholding with U channel)", img_otsu_second)
cv2.waitKey(0)
cv2.destroyAllWindows()

text_second = pytesseract.image_to_string(img_otsu_second, config='--oem 3 --psm 6 -l kor')
text_second = text_second.replace(" ", "")
text_second = text_second.replace("\n", "")
print("OCR 결과 (두 번째 이진화 후):", text_second)