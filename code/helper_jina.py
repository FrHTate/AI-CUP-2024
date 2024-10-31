import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm


model = AutoModelForSequenceClassification.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    torch_dtype="auto",
    trust_remote_code=True,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def faq_to_df(file):
    with open(file, "r") as f:
        data = json.load(f)
    faq_list = []
    for id, contents in data.items():
        for content in contents:
            content["doc_id"] = id
            for answer in content["answers"]:
                faq_list.append(
                    {
                        "id": int(id),
                        "text": content["question"] + answer,
                    }
                )
    return pd.DataFrame(faq_list)


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
            index = data[i]["index"]
            for j in range(0, len(text) - truncate_size + 1, truncate_size - overlap):
                segments.append(text[j : j + truncate_size])
                id.append(int(index))
            segments.append(label)
            id.append(int(index))
        return pd.DataFrame({"id": id, "text": segments}), category
    elif category == "ground_truths":
        df = pd.DataFrame(data)
        return df
    else:
        raise ValueError("Invalid category")


def compute_reranker_accuracy(
    insurance_path,
    finance_path,
    faq_path,
    truncate_size_i=256,
    overlap_i=32,
    truncate_size_f=256,
    overlap_f=32,
):
    # Convert JSON to DataFrame with text and id columns for different categories
    df_insurance, _ = json_to_df(insurance_path, truncate_size_i, overlap_f)
    df_finance, _ = json_to_df(finance_path, truncate_size_f, overlap_i)
    df_faq = faq_to_df(faq_path)

    gt_path = "/home/S113062628/project/AI-CUP-2024/dataset/preliminary/ground_truths_example.json"
    df_ground_truth = json_to_df(gt_path)
    ground_truth = df_ground_truth["retrieve"].tolist()

    # Load query JSON
    with open(
        "/home/S113062628/project/AI-CUP-2024/dataset/preliminary/questions_example.json"
    ) as f:
        queries = json.load(f)

    queries = queries["questions"]

    # Split queries into categories
    insurance_queries = queries[:50]
    finance_queries = queries[50:100]
    faq_queries = queries[100:150]

    insurance_gt = ground_truth[:50]
    finance_gt = ground_truth[50:100]
    faq_gt = ground_truth[100:150]

    categories = [
        (insurance_queries, insurance_gt, df_insurance, "Insurance"),
        (finance_queries, finance_gt, df_finance, "Finance"),
        (faq_queries, faq_gt, df_faq, "FAQ"),
    ]

    total_correct = 0
    total_queries = 0
    mismatches = []

    for category_queries, category_gt, df, category_name in categories:
        correct = 0
        total = len(category_queries)

        # Iterate through each query with tqdm for progress visualization
        for i, query in enumerate(
            tqdm(category_queries, desc=f"Processing Queries for {category_name}")
        ):
            query_text = query["query"]
            source_ids = set(query["source"])

            # Filter the passages based on source IDs
            relevant_passages = df[df["id"].isin(source_ids)]

            # If there are no relevant passages, skip the query
            if relevant_passages.empty:
                continue

            # Create sentence pairs between the query and each relevant passage
            sentence_pairs = [
                [query_text, passage] for passage in relevant_passages["text"]
            ]

            # Compute similarity scores for each sentence pair
            scores = model.compute_score(sentence_pairs, max_length=1024)

            # Convert the list of scores to a tensor
            scores_tensor = torch.tensor(scores, device=device)

            # Find the passage with the highest score
            best_match_idx = torch.argmax(scores_tensor).item()
            best_match_id = relevant_passages.iloc[best_match_idx]["id"]

            # Compare the best match ID with the ground truth
            if best_match_id == category_gt[i]:
                correct += 1
            else:
                mismatches.append(
                    {
                        "qid": total_queries + i + 1,
                        "query": query_text,
                        "predicted": int(best_match_id),
                        "ground_truth": int(category_gt[i]),
                    }
                )
        # Print individual accuracy for the category
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy for {category_name}: {accuracy * 100:.2f}%")

        total_correct += correct
        total_queries += total

    # Print total accuracy
    total_accuracy = total_correct / total_queries if total_queries > 0 else 0
    print(f"Total Accuracy: {total_accuracy * 100:.2f}%")

    with open("mismatch.json", "w") as f:
        json.dump(mismatches, f, ensure_ascii=False, indent=4)
