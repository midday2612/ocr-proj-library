import cv2

def convert_to_grayscale_yuv(image_path):
    # 이미지를 읽어옴
    image = cv2.imread(image_path)

    # 이미지를 YUV로 변환
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # U값만 가져와서 흑백 이미지로 변환
    u_channel = yuv_image[:,:,1]

    return u_channel

def apply_otsu_thresholding(image):
    # 오츠의 이진화 적용
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image

# 이미지 경로 설정
image_path = r"data/ttest.png"

# U값으로 변환된 이미지 가져오기
u_channel_image = convert_to_grayscale_yuv(image_path)

# 오츠의 이진화 적용
binary_image = apply_otsu_thresholding(u_channel_image)

# 변환된 이미지를 화면에 출력
cv2.imshow("Binary Image (Otsu's Thresholding with U channel)", binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
