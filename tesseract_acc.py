import os
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import pytesseract
from PIL import ImageFont, ImageDraw, Image
import json
import Levenshtein as lev

# 경로 설정
image_folder_path = "test"
output_file_path = "tesseract_acc_result.txt"  # 결과를 저장할 파일 경로
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def read_book_info_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        book_info = json.load(file)
    return book_info

# 'book_info.json' 파일을 읽어서 책 정보를 가져옴
book_info = read_book_info_from_json('book_info.json')

# Function to read text from an image using Tesseract OCR
def textRead(image):
    # Apply Tesseract OCR to recognize text
    config = ("-l kor --oem 3 --psm 7")
    text = pytesseract.image_to_string(image, config=config)
    
    return text.strip()

def search_key_in_dict(file_name, dictionary):
    for key in dictionary.keys():
        if key == file_name:
            return key
    return None

def find_most_similar_text(extracted_text, book_info):
    most_similar_text = None
    max_similarity = 0
    
    for key, value in book_info.items():
        similarity = lev.ratio(extracted_text.lower(), value.lower())
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_text = value
    
    if most_similar_text is None:
        return None, 0  # 유사한 텍스트가 없는 경우 None과 유사도 0을 반환
    else:
        return most_similar_text, max_similarity

def perform_text_detection_ocr(folder_path):
    # Get all image file paths in the specified folder
    def get_image_paths(folder_path):
        image_paths = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                image_paths.append(os.path.join(folder_path, filename))
        return image_paths

    # Get all image paths
    image_paths = get_image_paths(folder_path)
    count = 0
    total_count = 0
    total_similarity = 0
    
    # 결과를 저장할 파일 열기
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # Loop over each image in the folder
        for image_path in image_paths:
            # try:
            # Load the input image
            image = cv2.imread(image_path)
            # 파일 이름 검색
            file_name = os.path.basename(image_path)
            key = search_key_in_dict(file_name, book_info)
            if key:
                output_file.write("============================================================\n")
                output_file.write(f"파일 이름: {file_name}, 키: {key}, 책 제목: {book_info[key]}\n")
                output_file.write("============================================================\n")
                
                print("============================================================\n")
                print(f"파일 이름: {file_name}, 키: {key}, 책 제목: {book_info[key]}\n")
                print("============================================================\n")

                total_count += 1

            else:
                output_file.write(f"파일 이름: {file_name}, 해당하는 키 값을 찾을 수 없습니다.\n")
                print(f"파일 이름: {file_name}, 해당하는 키 값을 찾을 수 없습니다.\n")


            result = pytesseract.image_to_string(image, config='--oem 3 --psm 6 -l kor')

            result = result.replace(" ", "")
            result = result.replace("\n", "")

            output_file.write("result : " + result + "\n")
            print("result : " + result + "\n")
            most_similar, distance = find_most_similar_text(result, book_info)

            if most_similar is not None:
                output_file.write("최종결과 : " + most_similar + "\n")
                print("최종결과 : " + most_similar + "\n")
            else:
                output_file.write("최종결과 : 해당하는 책 제목이 없습니다.\n")
                print("최종결과 : 해당하는 책 제목이 없습니다.\n")

            output_file.write("유사도 : "+ str(distance) + "\n")
            print("유사도 : " + str(distance) + "\n")

            if most_similar == book_info[key]:
                count += 1
            output_file.write("현재 맞은 개수 : " + str(count) + "\n\n")
            print("현재 맞은 개수 : " + str(count) + "\n\n")
            
            total_similarity += distance
            
            # except Exception as e:
            #     output_file.write(f"이미지 처리 중 오류 발생: {e}\n")
            #     continue  # 오류 발생 시 해당 이미지를 건너뜁니다.

        accuracy = count / total_count if total_count > 0 else 0
        avg_similarity = total_similarity / total_count if total_count > 0 else 0
        
        output_file.write("최종 정확도 : " + str(accuracy) + "\n")
        output_file.write("평균 유사도 : " + str(avg_similarity) + "\n")
        
        print("최종 정확도 : " + str(accuracy) + "\n")
        print("평균 유사도 : " + str(avg_similarity) + "\n")

# Perform text detection and OCR on all images in the folder
perform_text_detection_ocr(image_folder_path)
