import cv2
import numpy as np
import pytesseract
import Levenshtein as lev
from PIL import Image

# 화이트 스페이스 제거
def remove_spaces_and_whitespace(input_list):
    cleaned_list = [element.replace(" ", "").strip() for element in input_list if isinstance(element, str)]
    return cleaned_list

# 데이터베이스와 비교하여 유사도가 높은 값 찾기
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
    if (highest_similarity >= 0.2):
        result = closest_match
    else:
        # result = closest_match
        result = "판정 불가"

    return result

# 4개의 점을 기준으로 사각형 모양으로 이미지 자르기
def crop_image_with_coordinates(image, coordinates):

    pts = np.array(coordinates, np.int32)
    pts = pts.reshape((-1,1,2))
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], (255,255,255))
    masked_image = cv2.bitwise_and(image, mask)
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    cropped_image = masked_image[y:y+h, x:x+w].copy()

    return cropped_image


# Tesseract OCR 실행 파일 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 텍스트 파일 경로 설정
input_file_path = r"data\resource_names.txt"

# 텍스트 파일에서 데이터 읽어오기
with open(input_file_path, 'r', encoding='utf-8') as file:
    resource_names_list = [line.strip() for line in file.readlines()]
resource_names_list = remove_spaces_and_whitespace(resource_names_list)

# 이미지 파일 읽기
image = cv2.imread(r'data\a1.jpg')

# 이미지 사이즈 조정
scale_percent = 30
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
image = cv2.resize(image, (width, height))

# 그레이스케일 이미지로 변환
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 가우시안 블러 필터 적용
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Canny 엣지 검출
edges = cv2.Canny(blurred_image, 10, 10, apertureSize=3)
'''
10: 엣지 검출의 최소 임계값
10: 엣지 검출의 최대 임계값 - 이 이상 -> 확실한 엣지로 간주
apertureSize = 3 : 소벨 커널의 크기 기본값은 3, 작은 값 -> 더 세밀한 엣지, 노이즈 민감 // 큰 값 -> 더 엣지 무시
'''


# 허프 변환을 적용하여 직선 검출
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=220) #threshold=220가 제이 나은 듯


'''
거리 해상도(rho) -> 1픽셀 단위로 해상도 설정
각도 해상도(theta) -> 1도 단위로 해상도 설정
threshold=220 : 라인으로 판단되기 위한 최소 갯수 -> 선의 수 조절 가능
'''

if lines is not None:
    upper_line_x = []
    under_line_x = []

    splitted_images = []


    for line in lines:
        for rho, theta in line:
            
            # 수직선에 가까운지 확인
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 100 * (-b))
            y1 = int(y0 + 100 * (a))+image.shape[0]
            x2 = int(x0 - 100 * (-b))
            y2 = int(y0 - 100 * (a))+image.shape[0]
            
            # 선의 각도가 수직선에 가까운지 확인 (예: 80도에서 100도 사이)
            theta_degree = np.degrees(theta)
            if ((80+90 <= theta_degree <= 100+90) or (260+90 <= theta_degree <= 280+90)):
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                upper_line_x.append(x1)
                under_line_x.append(x2)
                
                # # 선의 길이 계산
                # line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
                # # 선의 길이가 150 이상인 경우에만 그리기
                # if line_length >= 150:
                #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                #     upper_line_x.append(x1)
                #     under_line_x.append(x2)

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
        # if (under_line_x[i] < 0):
        #     under_line_x[i] = 0
        # if (upper_line_x[i] < 0):
        #     upper_line_x[i] = 0

        coordinates = [upper_line_x[i], 0, upper_line_x[i+1], 0, under_line_x[i+1], image.shape[0], under_line_x[i], image.shape[0]]
        if(upper_line_x[i+1]-upper_line_x[i]>50):
            cropped_image = crop_image_with_coordinates(image, coordinates)
        
            text = pytesseract.image_to_string(cropped_image, config='--oem 3 --psm 6 -l kor')
            text = text.replace(" ", "")
            text = text.replace("\n", "")
            print("----------------------------------------")
            print(f"<segment {i+1}>")
            if text:
                print("ocr 결과 : ", text)
                print("가장 유사한 제목 : ", find_closest_match(text, resource_names_list))
            else:
                print("인식된 텍스트 없음")
            
            cv2.imshow(f'Segment {i}', cropped_image)
            cv2.waitKey(0)  
            cv2.destroyAllWindows() 

'''
#해결 과제
1. 적절한 임계값 찾기
    -이미지마다 적절한 임계값이 다름
2. 눕혀진 텍스트 구분 어떻게?
    -학습 시켜서?
    -90도씩 돌려가면서 유사도 제일 높게 나온 값으로? - 계산비용 너무 높음 - 일단 해보기
3. kor+eng 하면 정확도 너무 낮아짐
    -테서랙트 말고 다른 거?
    -전처리 잘 하면 나아질 수도 있음ㅌ
'''