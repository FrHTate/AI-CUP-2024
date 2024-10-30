# Import necessary libraries
import os
import json
import pandas as pd
import numpy as np
import csv
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from collections import Counter

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

    if category == "questions":
        df = pd.DataFrame(data)
        df = df.rename(columns={"query": "text"})
        return df, category

    elif category == "finance" or category == "insurance":
        for i in range(len(data)):
            text = data[i]["text"]
            label = data[i]["label"]
            for j in range(0, len(text) - truncate_size + 1, truncate_size - overlap):
                segments.append(text[j : j + truncate_size])
                id.append(i)
            segments.append(label)
            id.append(i)
        return pd.DataFrame({"id": id, "text": segments}), category
    else:
        raise ValueError("Invalid category")


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
    with open(f"{category}_embeddings.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "embeddings"])  # Write header
        for _, row in tqdm(
            output_df.iterrows(), total=output_df.shape[0], desc="Saving to CSV"
        ):
            writer.writerow([row["id"], row["embeddings"]])


def query_embedding(file_path, truncate_size=32, overlap=8):
    file_df, _ = json_to_df(file_path, truncate_size, overlap)
    embeddings = []
    for query in tqdm(file_df["text"], desc="Embedding query"):
        input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(input_ids)
        embeddings.append(output.logits[:, 0, :].squeeze().cpu().numpy().tolist())
    file_df["embeddings"] = embeddings
    with open(f"query_embeddings.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(file_df.columns)

        for row in tqdm(
            file_df.itertuples(index=False, name=None),
            desc="Saving to CSV",
            total=len(file_df),
        ):
            writer.writerow(row)


# Retrive most related file to the query
def accuracy_of_category(ground_truth, answer):
    match = sum((Counter(answer) & Counter(ground_truth)).values())
    total = sum(Counter(ground_truth).values())
    return match / total


def retriever(query_path, passage_path):
    # Load the data with progress visualization
    query_df = pd.read_csv(query_path)
    passage_df = pd.read_csv(passage_path)

    # Progress bar for loading embeddings column into lists
    query_df["embeddings"] = [
        eval(embedding)
        for embedding in tqdm(query_df["embeddings"], desc="Loading query embeddings")
    ]
    passage_df["embeddings"] = [
        eval(embedding)
        for embedding in tqdm(
            passage_df["embeddings"], desc="Loading passage embeddings"
        )
    ]

    match_results = []

    # Iterate through each query
    for _, query_row in tqdm(
        query_df.iterrows(), desc="Processing queries", total=len(query_df)
    ):
        query_embedding = [query_row["embeddings"]]
        source_ids = eval(
            query_row["source"]
        )  # Convert source column string to list of IDs

        # Filter passages by source IDs
        relevant_passages = passage_df[passage_df["id"].isin(source_ids)]
        passage_embeddings = relevant_passages["embeddings"].tolist()

        # Calculate cosine similarity between the query and relevant passages
        if passage_embeddings:
            similarities = cosine_similarity(query_embedding, passage_embeddings)[0]
            best_match_id = relevant_passages.iloc[similarities.argmax()]["id"]
        else:
            best_match_id = None  # No relevant passages found

        match_results.append(best_match_id)

    return match_results
