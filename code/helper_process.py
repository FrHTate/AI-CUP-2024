# Import necessary libraries
import os
import json
import pandas as pd
import numpy as np
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
    category = list(data.keys())[0]
    data = data[category]

    segments = []
    id = []

    for i in range(len(data)):
        text = data[i]["text"]
        for j in range(0, len(text) - truncate_size + 1, truncate_size - overlap):
            segments.append(text[j : j + truncate_size])
            id.append(i)

    return pd.DataFrame({"id": id, "text": segments}), category


# Function to embed passages using the model
def passage_embedding(file_path, truncate_size=128, overlap=32):
    file_df, category = json_to_df(file_path, truncate_size, overlap)
    embeddings = []
    for text in tqdm(file_df["text"], desc=f"Embedding {category}"):
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(input_ids)
        embeddings.append(output.logits[:, 0, :].squeeze().cpu().numpy().tolist())
    output_df = pd.DataFrame({"id": file_df["id"], "embeddings": embeddings})
    print(f"Saving {category} embeddings to {category}_embeddings.csv")
    output_df.to_csv(f"{category}_embeddings.csv", index=False)
