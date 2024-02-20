import cv2
import numpy as np
def draw_yellow_rectangles(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    original_image = image.copy()  # 원본 이미지 복사
    
    # BGR에서 HSV로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 노란색 범위 정의 (HSV)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # 노란색에 해당하는 영역 찾기
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 노란색 영역의 외곽선 찾기
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 각각의 외곽선을 감싸는 최소 크기의 직사각형 그리기
    for contour in contours:
        # 각 외곽선을 감싸는 최소 크기의 직사각형 구하기
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    # 결과 이미지 출력
    cv2.imshow('Yellow Rectangles', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_yellow_rectangles(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    original_image = image.copy()  # 원본 이미지 복사
    image = cv2.imread(image_path)
    scale_percent = 30
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height))
    original_image = cv2.resize(original_image, (width, height))
    # BGR에서 HSV로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 노란색 범위 정의 (HSV)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # 노란색에 해당하는 영역 찾기
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 노란색 영역의 외곽선 찾기
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imshow('Yellow Rectangles', yellow_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 사각형에 가까운 외곽선만 남기기
    min_rect_width = 5  # 최소 너비
    min_rect_height = 10  # 최소 높이
    filtered_contours = []
    for contour in contours:
        # 외곽선을 근사화하여 근사 다각형으로 변환
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 근사 다각형이 사각형 조건을 만족하는지 확인
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio >= 0.5 and aspect_ratio <= 2.0 and w >= min_rect_width and h >= min_rect_height:
                filtered_contours.append(contour)

    # 각각의 외곽선을 감싸는 사각형 그리기
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    # 결과 이미지 출력
    cv2.imshow('Yellow Rectangles', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# 이미지 경로 지정
image_path = r'data\a1.jpg'



# 함수 호출
# draw_yellow_contours(image_path)
draw_yellow_rectangles(image_path)
