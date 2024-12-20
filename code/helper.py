import os
import pandas as pd
from bm25_retrieve import read_pdf
from pdf2image import convert_from_path
import pytesseract
from multiprocessing import Pool
from tqdm import tqdm
import json
from image_preprocessor import pdf_image_label_extractor, pdf_image_whole_file


# extract text from pdf image
def read_pdf_image(path):
    pages = convert_from_path(path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang="chi_tra")

    return text


def preprocessor(path):
    reference = ["finance"]

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
                labels.append(pdf_image_label_extractor(file_path, dpi=400)[:50])
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


# OCR preprocessor
def process_file_ocr(file_path):
    # print(f"processing {file_path}")
    text = pdf_image_whole_file(file_path, dpi=400)
    text = text.replace("\n", "").replace("\r", "").replace("\f", "").replace(" ", "")

    text = text[50:]
    label = text[:50]
    idx = os.path.basename(file_path)[:-4]  # 更改變數名稱
    return idx, label, text


def preprocessor_ocr(path):
    reference = ["finance"]

    for ref in reference:
        ref_path = os.path.join(path, ref)
        pdf_files = [
            os.path.join(ref_path, file)
            for file in os.listdir(ref_path)
            if file.endswith(".pdf")
        ]

        # 確保輸出目錄存在
        output_dir = "./reference"
        os.makedirs(output_dir, exist_ok=True)

        with Pool(12) as pool:
            results = []
            for result in tqdm(
                pool.imap_unordered(process_file_ocr, pdf_files),
                total=len(pdf_files),
                desc=f"Processing {ref} files",
            ):
                results.append(result)

        data = [{"index": idx, "label": lbl, "text": txt} for idx, lbl, txt in results]
        with open(f"{output_dir}/ocr_{ref}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Test Preprocessor
    preprocessor("./reference")
    preprocessor_ocr("./reference")
