from pdf2image import convert_from_path
from PIL import ImageFilter
import pytesseract
import json


def remove_color(image, threshold=130):
    image = image.convert("RGBA")
    data = image.getdata()
    new_data = []
    for item in data:
        if item[0] > threshold:
            new_data.append((255, 255, 255, item[3]))  # 替換為白色
        else:
            new_data.append(item)  # 保留原來的顏色
    image.putdata(new_data)
    # 將圖片轉換為 RGB 以儲存為 JPEG
    image = image.convert("RGB")
    return image


def extract_text(image):
    text = pytesseract.image_to_string(image, lang="chi_tra", config="--psm 6")
    text = text.replace("\n", "").replace(" ", "")
    return text


def pdf_image_label_extractor(path, dpi=300):
    # 讀取 PDF 檔案，取得第一頁的圖片
    pages = convert_from_path(path, dpi=dpi)
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


def pdf_image_whole_file(path, dpi=300):
    pages = convert_from_path(path, dpi=dpi)
    text = str()
    for i, image in enumerate(pages):
        image = remove_color(image)
        # image.save(f"{path[:-4]}_{i}.jpg")
        text += extract_text(image)
        # print(text)
    return text


if __name__ == "__main__":
    # test function
    # 抓image pdf的label
    text = pdf_image_label_extractor("300.pdf")
    print(text)

    # 抓image pdf的全部文字
    text = pdf_image_whole_file("757.pdf", dpi=400)
    print(text)
