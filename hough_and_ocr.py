import cv2
import numpy as np
import pytesseract
import Levenshtein as lev

def remove_spaces_and_whitespace(input_list):
    cleaned_list = [element.replace(" ", "").strip() for element in input_list if isinstance(element, str)]
    return cleaned_list

def find_closest_match(ocr_result, database):

    closest_match = None
    highest_similarity = 0

    for title in database:
        similarity = lev.ratio(ocr_result.lower(), title.lower())
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_match = title
    #         print(similarity)
    # print("high= ", highest_similarity)
    # print(closest_match)
    if (highest_similarity >= 0.1):
        result = closest_match
    else:
        # result = closest_match
        result = "판정 불가"

    return result

def crop_image_with_coordinates(image, coordinates):
    """
    이미지에서 주어진 좌표에 해당하는 영역을 자릅니다.
    
    Args:
    - image: 원본 이미지
    - coordinates: 4개의 좌표 [x1, y1, x2, y2, x3, y3, x4, y4] (좌측 상단부터 시계 방향으로)
    
    Returns:
    - cropped_image: 잘린 영역 이미지
    """
    pts = np.array(coordinates, np.int32)
    pts = pts.reshape((-1,1,2))
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], (255,255,255))
    masked_image = cv2.bitwise_and(image, mask)
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    cropped_image = masked_image[y:y+h, x:x+w].copy()

    return cropped_image

    # cv2.imshow('cropped_image', cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

# Tesseract OCR 실행 파일 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 텍스트 파일 경로 설정
input_file_path = r"data\resource_names.txt"

# 텍스트 파일에서 데이터 읽어오기
with open(input_file_path, 'r', encoding='utf-8') as file:
    resource_names_list = [line.strip() for line in file.readlines()]
resource_names_list = remove_spaces_and_whitespace(resource_names_list)

# 이미지 파일 읽기
image = cv2.imread(r'data\lib2.jpg')

# 이미지 사이즈 조정
scale_percent = 30  # 이미지 크기를 줄일 비율 (원본의 50%로 줄임)
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
image = cv2.resize(image, (width, height))

# 그레이스케일 이미지로 변환
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 가우시안 블러 필터 적용
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Canny 엣지 검출
edges = cv2.Canny(blurred_image, 50, 10, apertureSize=3)

# 허프 변환을 적용하여 직선 검출
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=220)

if lines is not None:
    upper_line_x = []
    under_line_x = []

    splitted_images = []
    vertical_lines_x = []
    line_endpoints = []

    for line in lines:
        for rho, theta in line:
            
            # 수직선에 가까운지 확인
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # 선을 기준으로 분할하기 위한 시작점과 끝점 저장
            line_endpoints.append((x1, y1, x2, y2))
            
            # 선의 각도가 수직선에 가까운지 확인 (예: 80도에서 100도 사이)
            theta_degree = np.degrees(theta)
            if ((80+90 <= theta_degree <= 100+90) or (260+90 <= theta_degree <= 280+90)):
            # if abs(theta_degree - 90) < 10 or abs(theta_degree - 270) < 10:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                vertical_lines_x.append(x0)

                # 선의 길이 계산
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
                # 선의 길이가 150 이상인 경우에만 그리기
                if line_length >= 5:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    upper_line_x.append(x1)
                    under_line_x.append(x2)

    upper_line_x.insert(0, 0)
    under_line_x.insert(0, 0)

    upper_line_x.sort()
    under_line_x.sort()

    print(upper_line_x)
    print(under_line_x)


    # 결과 이미지를 화면에 표시
    cv2.imshow('Hough Line Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    for i in range(len(upper_line_x)-1):
        if (under_line_x[i] < 0):
            under_line_x[i] = 0
        if (upper_line_x[i] < 0):
            upper_line_x[i] = 0

        coordinates = [upper_line_x[i], 0, upper_line_x[i+1], 0, under_line_x[i+1], image.shape[0], under_line_x[i], image.shape[0]]
        if(upper_line_x[i+1]-upper_line_x[i]>30):
            cropped_image = crop_image_with_coordinates(image, coordinates)
        
            kernel = np.ones((1, 1), np.uint8)
            eroded_image = cv2.erode(crop_image_with_coordinates(image, coordinates), kernel, iterations=3)

            # cv2.imshow(f'Segment {i}', eroded_image)
            # cv2.waitKey(0)  # 사용자가 키를 누를 때까지 
            # cv2.destroyAllWindows()  # 모든 창 닫기

            text = pytesseract.image_to_string(eroded_image, config='--oem 3 --psm 6 -l kor')
            text = text.replace(" ", "")
            text = text.replace("\n", "")
            if text:
                print("ocr 결과 : ", text)
                print("가장 유사한 제목 : ", find_closest_match(text, resource_names_list))
                print("\n")
            else:
                print("인식된 텍스트 없음")
