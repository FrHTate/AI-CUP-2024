<<<<<<< HEAD
# Import necessary libraries
import os
import json
import pandas as pd
=======
import os
import pandas as pd
from bm25_retrieve import read_pdf
>>>>>>> refs/remotes/origin/main
from pdf2image import convert_from_path
import pytesseract
<<<<<<< HEAD
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

# Load tokenizer and model for Chinese text processing
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model.to(device)
=======
from multiprocessing import Pool
from tqdm import tqdm
import json
>>>>>>> refs/remotes/origin/main


# Function to extract text from PDF images using OCR
def read_pdf_image(path):
    pages = convert_from_path(path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang="chi_tra")  # Chinese Traditional
    return text


<<<<<<< HEAD
# Example usage of PDF text extraction
path = "./reference/finance/5.pdf"
print(read_pdf_image(path))


# Function to load FAQ data from JSON and convert it into a DataFrame
def load_faq_as_df(file):
    with open(file, "r") as f:
        data = json.load(f)
    faq_list = []
    for doc_id, contents in data.items():
        for content in contents:
            content["doc_id"] = doc_id
            for answer in content["answers"]:
                faq_list.append(
                    {
                        "doc_id": doc_id,
                        "question": content["question"],
                        "answer": answer,
                    }
                )
    return pd.DataFrame(faq_list)


# Function to load JSON data, segment it, and return as DataFrame
def json_to_df(file_path, truncate_size=128, overlap=32):
    with open(file_path, "r") as f:
        data = json.load(f)
        document_name = os.path.basename(f.name).split(".")[0]

    # Preprocess text and segment it
    data = data["text"].replace("\n", "")
    segments = [
        data[i : i + truncate_size]
        for i in range(0, len(data) - truncate_size + 1, truncate_size - overlap)
    ]

    # Return DataFrame with document name and segmented text
    return pd.DataFrame({"document_name": document_name, "text": segments})


# Function to embed passages using the model
def passage_embedding(file_path, truncate_size=128, overlap=32):
    file_df = json_to_df(file_path, truncate_size, overlap)
    embeddings = []
    for text in file_df["text"]:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(input_ids)
        embeddings.append(output.logits[:, 0, :].squeeze())  # Keep embeddings on GPU
    return torch.stack(embeddings)


# Function to process files in a source directory and build embeddings
def data_loader(source_path):
    files = os.listdir(source_path)
    for file in tqdm(files, desc="Building embeddings"):
        file_path = os.path.join(source_path, file)
        embeddings = passage_embedding(file_path)
        print(embeddings)
=======
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
        texts = []

        pdf_files = [file for file in os.listdir(ref_path) if file.endswith(".pdf")]
        for file_name in tqdm(pdf_files, desc=f"Processing {ref} files"):
            file_path = os.path.join(ref_path, file_name)
            text = read_pdf(file_path)
            if text.strip() == "":
                text = read_pdf_image(file_path)

            # Remove characters that might interfere with JSON parsing
            text = text.replace("\n", "").replace("\r", "").strip()
            index.append(file_name[:-4])
            texts.append(text)

        data = [{"index": idx, "text": txt} for idx, txt in zip(index, texts)]
        with open(f"./reference/{ref}.json", "w", encoding="utf-8") as f:
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
>>>>>>> refs/remotes/origin/main
