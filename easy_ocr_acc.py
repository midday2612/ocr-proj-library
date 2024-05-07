import os
from imutils.object_detection import non_max_suppression
import cv2
import easyocr
import json
import Levenshtein as lev

# 경로 설정
image_folder_path = "test"
output_file_path = "easy_ocr_acc_result.txt"  # 결과를 저장할 파일 경로

# 설정 값
min_confidence = 0.5
width = 320
height = 320

reader = easyocr.Reader(['ko'])

# 'book_info.json' 파일을 읽어서 책 정보를 가져옴
def read_book_info_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        book_info = json.load(file)
    return book_info

# 'book_info.json' 파일을 읽어서 책 정보를 가져옴
book_info = read_book_info_from_json('book_info.json')

# Function to read text from an image using EasyOCR
def textRead(image):
    # Initialize EasyOCR reader with Korean as the language
    reader = easyocr.Reader(['ko'])
    # Recognize text using EasyOCR
    result = reader.readtext(image)
    # Combine recognized text into a single string
    # text = ' '.join([text for text, _, _ in result])
    
    return result

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
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # Get all image paths
        image_paths = get_image_paths(folder_path)
        count = 0
        total_num = 0
        total_similarity = 0
        # Loop over each image in the folder
        for image_path in image_paths:
            try:
                # Load the input image
                image = cv2.imread(image_path)
                # 파일 이름 검색
                file_name = os.path.basename(image_path)
                key = search_key_in_dict(file_name, book_info)
                if key:
                    print("============================================================")
                    print(f"파일 이름: {file_name}, 키: {key}, 책 제목: {book_info[key]}")
                    print("============================================================")
                    output_file.write("============================================================\n")
                    output_file.write(f"파일 이름: {file_name}, 키: {key}, 책 제목: {book_info[key]}\n")
                    output_file.write("============================================================\n")
                    total_num += 1
                else:
                    print(f"파일 이름: {file_name}, 해당하는 키 값을 찾을 수 없습니다.")
                    output_file.write(f"파일 이름: {file_name}, 해당하는 키 값을 찾을 수 없습니다.")

                results = reader.readtext(image_path)
                
                extracted_texts = [text for _, text, _ in results]
                # print(extracted_texts)
                # Combine recognized text into a single string
                result = " ".join(extracted_texts)
                print("ocr 결과 : ", result)
                output_file.write("ocr 결과 : "+ result +"\n")
                # Find the most similar text in book_info.json
                # most_similar_text = find_most_similar_text(result, book_info)
                most_similar_text, distance = find_most_similar_text(result, book_info)
                print("가장 비슷한 책 제목 : ", most_similar_text)
                print("유사도 : ", distance)
                output_file.write("가장 비슷한 책 제목 : "+ most_similar_text + "\n")
                output_file.write("유사도 : "+ str(distance)+ "\n")

                # print("최종결과 : ", find_most_similar_text(result, book_info)[0])
                if (most_similar_text==book_info[key]):
                    count += 1
                print("현재 맞은 개수 : ", count, "\n")
                output_file.write("현재 맞은 개수 : "+ str(count)+ "\n")
                total_similarity += distance

            except Exception as e:
                print(f"이미지 처리 중 오류 발생: {e}")
                output_file.write(f"이미지 처리 중 오류 발생: {e}"+"\n")
                continue  # 오류 발생 시 해당 이미지를 건너뜁니다.
        
    accuracy = count / total_num if total_num > 0 else 0
    avg_similarity = total_similarity / total_num if total_num > 0 else 0
    
    output_file.write("최종 정확도 : " + str(accuracy) + "\n")
    output_file.write("평균 유사도 : " + str(avg_similarity) + "\n")
    
    print("최종 정확도 : " + str(accuracy) + "\n")
    print("평균 유사도 : " + str(avg_similarity) + "\n")
    # print("최종 정확도 : ", count/total_num)
    # output_file.write("최종 정확도 : "+ str(count/total_num))

# Perform text detection and OCR on all images in the folder
perform_text_detection_ocr(image_folder_path)
