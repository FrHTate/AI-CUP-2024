# Import necessary libraries
import os
import json
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

# Load tokenizer and model for Chinese text processing
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model.to(device)


# Function to extract text from PDF images using OCR
def read_pdf_image(path):
    pages = convert_from_path(path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang="chi_tra")  # Chinese Traditional
    return text


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
