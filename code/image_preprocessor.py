from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import json


def remove_color(image):
    image = image.convert("RGBA")
    data = image.getdata()
    new_data = []
    for item in data:
        # 如果 R 像素值大於 150，則替換為白色(因為大多不要的顏色都是紅色)
        if item[0] > 150:
            new_data.append((255, 255, 255, item[3]))  # 替換為白色
        else:
            new_data.append(item)  # 保留原來的顏色
    image.putdata(new_data)
    # 將圖片轉換為 RGB 以儲存為 JPEG
    image = image.convert("RGB")
    return image


def extract_text(image):
    text = pytesseract.image_to_string(image, lang="chi_tra")
    text = text.replace("\n", "").replace(" ", "")
    return text


def pdf_image_label_extractor(path):
    # 讀取 PDF 檔案，取得第一頁的圖片
    pages = convert_from_path(path)
    image = pages[0]
    # 裁切圖片
    width, height = image.size
    image = image.crop((0, 0, width, height // 4))
    # 去除顏色
    image.save(f"{path[:-4]}_origin.jpg")
    image = remove_color(image)
    image.save(f"{path[:-4]}_removed.jpg")
    text = extract_text(image)
    return text


if __name__ == "__main__":
    text = pdf_image_label_extractor("675.pdf")
    print(text)
