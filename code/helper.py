import os
import pandas as pd
from bm25_retrieve import read_pdf
from pdf2image import convert_from_path
import pytesseract
from multiprocessing import Pool
from tqdm import tqdm
import json
from image_preprocessor import pdf_image_label_extractor


# extract text from pdf image
def read_pdf_image(path):
    pages = convert_from_path(path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang="chi_tra")

    return text


def mark_category(file_path):
    text = read_pdf(file_path)
    if text.strip() == "":
        return "image", file_path[:-4]
    else:
        return "text", file_path[:-4]


# Only want to track how much image pdfs in reference folder
def track_image_pdf(path):
    reference = ["finance", "insurance"]
    image_idx = []
    text_idx = []

    for ref in reference:
        ref_path = os.path.join(path, ref)
        pdf_files = [
            os.path.join(ref_path, file)
            for file in os.listdir(ref_path)
            if file.endswith(".pdf")
        ]

        with Pool(os.cpu_count()) as pool:
            results = pool.map(mark_category, pdf_files)

        for result_type, file_path in results:
            file_name = os.path.basename(file_path)
            file_category = os.path.basename(os.path.dirname(file_path))
            if result_type == "image":
                image_idx.append([file_category, file_name])
            else:
                text_idx.append([file_category, file_name])

    return image_idx, text_idx


def preprocessor(path):
    reference = ["finance", "insurance"]

    for ref in reference:
        ref_path = os.path.join(path, ref)
        index = []
        labels = []
        texts = []
        i = 0
        pdf_files = [file for file in os.listdir(ref_path) if file.endswith(".pdf")]
        for file_name in tqdm(pdf_files, desc=f"Processing {ref} files"):
            file_path = os.path.join(ref_path, file_name)
            text = read_pdf(file_path)
            text = text.replace("\n", "").replace("\r", "").replace(" ", "")
            if text == "":
                text = read_pdf_image(file_path)
                text = text.replace("\n", "").replace("\r", "").replace(" ", "")
                labels.append(pdf_image_label_extractor(file_path)[:50])
            else:
                labels.append(text[:50])

            # Remove characters that might interfere with JSON parsing
            index.append(file_name[:-4])
            texts.append(text)

            # i += 1
            # if i == 10:
            #     break

        data = [
            {"index": idx, "label": lbl, "text": txt}
            for idx, lbl, txt in zip(index, labels, texts)
        ]
        with open(f"./reference/{ref}_test.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Test read_pdf_image
    # path = "./reference/finance/1.pdf"
    # print(read_pdf_image(path))

    # Test track_image_pdf
    # image_idx, text_idx = track_image_pdf("./reference")
    # print(image_idx)

    # Test Preprocessor
    preprocessor("./reference")
    pass
