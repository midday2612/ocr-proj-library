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
edges = cv2.Canny(blurred_image, 10, 10, apertureSize=3)

# 허프 변환을 적용하여 직선 검출
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=220)
##############################################################
# if lines is not None:
#     # 분할된 이미지 저장을 위한 리스트
#     splitted_images = []
#     # 세로선의 x좌표를 저장하기 위한 리스트
#     vertical_lines_x = []
#     line_endpoints = []
#     for line in lines:
#         for rho, theta in line:
            
#             # 수직선에 가까운지 확인
           
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             x1 = int(x0 + 1000 * (-b))
#             y1 = int(y0 + 1000 * (a))
#             x2 = int(x0 - 1000 * (-b))
#             y2 = int(y0 - 1000 * (a))

#             # 선을 기준으로 분할하기 위한 시작점과 끝점 저장
#             line_endpoints.append((x1, y1, x2, y2))
            
#             # 선의 각도가 수직선에 가까운지 확인 (예: 80도에서 100도 사이)
#             theta_degree = np.degrees(theta)
#             if ((80+90 <= theta_degree <= 100+90) or (260+90 <= theta_degree <= 280+90)):
#             # if abs(theta_degree - 90) < 10 or abs(theta_degree - 270) < 10:
#                 cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 vertical_lines_x.append(x0)

#                 # 선의 길이 계산
#                 line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
#                 # 선의 길이가 150 이상인 경우에만 그리기
#                 if line_length >= 5:
#                     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     vertical_lines_x.append(x0)
#                     # 결과 이미지를 화면에 표시
#                     # cv2.imshow('Hough Line Detection', image)
#                     # cv2.waitKey(0)
#                     # cv2.destroyAllWindows()

            

#     vertical_lines_x.sort()

#     filtered_lines_x = [vertical_lines_x[0]]
#     for x in vertical_lines_x[1:]:
#         if x - filtered_lines_x[-1] >= 10:  # 거리가 10 이상인 경우에만 추가
#             filtered_lines_x.append(x)
    

#     # 결과 이미지를 화면에 표시
#     cv2.imshow('Hough Line Detection', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # 이미지 분할
#     prev_x = 0
#     for x in filtered_lines_x:
#         splitted_image = image[:, int(prev_x):int(x)]
#         splitted_images.append(splitted_image)
#         prev_x = x
#     # 마지막 세그먼트 추가
#     splitted_images.append(image[:, int(prev_x):])

#     for idx, img in enumerate(splitted_images):
#         # scale_percent = 200 # 이미지 크기를 줄일 비율 (원본의 50%로 줄임)
#         # width = int(img.shape[1] * scale_percent / 100)
#         # height = int(img.shape[0] * scale_percent / 100)
#         # img = cv2.resize(img, (width, height))

#         cv2.imshow(f'Segment {idx+1}', img)
#         cv2.waitKey(0)  # 사용자가 키를 누를 때까지 대기
#         cv2.destroyAllWindows()  # 모든 창 닫기
    
#     # 분할된 각 이미지에 대해 OCR 진행
#     for idx, img in enumerate(splitted_images):
#         text = pytesseract.image_to_string(img, config='--oem 3 --psm 6 -l kor')
#         text = text.replace(" ", "")
#         text = text.replace("\n", "")
#         if text:
#             print("ocr 결과 : ", text)
#             print("가장 유사한 제목 : ", find_closest_match(text, resource_names_list))