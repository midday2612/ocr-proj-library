import easyocr
import json
import Levenshtein as lev
import matplotlib.pyplot as plt
import os

filename = '002.jpg'
folder_path = r"test/"
file_path = os.path.join(folder_path, filename)
reader = easyocr.Reader(['ko'])
results = reader.readtext(file_path)

results_list = []

def bounding_box_size(box):
    x1, y1 = box[0]
    x2, y2 = box[2]
    return (x2 - x1) * (y2 - y1)

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
        results_list.append((most_similar_text, max_similarity))
        return most_similar_text, max_similarity

def search_title_in_dict(file_name, dictionary):
    if file_name in dictionary:
        return dictionary[file_name]
    else:
        return None
    
def read_book_info_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        book_info = json.load(file)
    return book_info

# 'book_info.json' 파일을 읽어서 책 정보를 가져옴
book_info = read_book_info_from_json('book_info.json')

# 바운딩 박스의 크기를 기준으로 정렬
sorted_indices = sorted(range(len(results)), key=lambda i: bounding_box_size(results[i][0]))

# OCR 결과 텍스트만을 저장할 리스트
ocr_texts = []

# OCR 결과에서 텍스트 부분만 추출하여 리스트에 저장
for result in results:
    ocr_texts.append(result[1])

# Filter out bounding boxes with low confidence
filtered_results = [result for result in results if result[2] > 0.5]  # Example threshold: 0.5

for sorted_index in sorted_indices:
    ocr_texts[sorted_index] = None
    output_text = ''.join([text for text in ocr_texts if text is not None])
    find_most_similar_text(output_text, book_info)

# Assuming you have `results_list` with similarity scores
# x-axis represents indices starting from 1
x = range(1, len(results_list) + 1)

# y-axis represents similarity scores
y = [score[1] for score in results_list]

color_list = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan']
title_list = []
title_list.append(results_list[0][0])
color_index = 0

title = search_title_in_dict(filename, book_info)

# Plotting the scatter plot
for score in results_list:
    if score[0] in title_list:  # Most similar text is not equal to the correct answer
        color_index = title_list.index(score[0])
    else:
        title_list.append(score[0])
        color_index = title_list.index(score[0])
    plt.scatter(x[results_list.index(score)], score[1], c=color_list[color_index % len(color_list)])  # Plotting red for incorrect matches
    plt.annotate(score[0], (x[results_list.index(score)], score[1]), textcoords="offset points", xytext=(0,10), ha='center',  fontname='NanumGothic',fontsize=6)  # Annotating incorrect matches with smaller font size
  
plt.xlabel('Index')
plt.ylabel('Similarity Score')
plt.title(f"{title}",fontname='NanumGothic',fontsize=12)
plt.grid(True)
plt.show()
