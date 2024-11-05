import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
import numpy as np
from tqdm import tqdm

# Load the initial reranker model
model = AutoModelForSequenceClassification.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    torch_dtype="auto",
    trust_remote_code=True,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device)

# Load the cross-encoder model for reranking
cross_encoder = CrossEncoder(
    "jinaai/jina-reranker-v2-base-multilingual",
    automodel_args={"torch_dtype": "auto"},
    trust_remote_code=True,
)


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


def json_to_df(file_path, chunk_size=128, overlap=32, summary=False):
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

    elif category == "insurance":
        for i in range(len(data)):
            index = data[i]["index"]
            text = data[i]["text"]
            label = data[i]["label"]
            if summary:
                summary_passage = data[i]["summary"]
                segments.append(summary_passage)
                id.append(int(index))
            # segments.append(label)
            # id.append(int(index))
            for j in range(0, len(text) - chunk_size + 1, chunk_size - overlap):
                segments.append(
                    text[j : j + chunk_size] + " [標題] " + label + " [/標題]."
                )
                id.append(int(index))
        return pd.DataFrame({"id": id, "text": segments}), category
    elif category == "finance":
        for i in range(len(data)):
            index = data[i]["index"]
            text = data[i]["text"]
            label = data[i]["label"]
            if summary:
                summary_passage = data[i]["summary"]
                segments.append(summary_passage)
                id.append(int(index))
            # if label != "":
            # segments.append(label)
            # id.append(int(index))
            for j in range(0, len(text) - chunk_size + 1, chunk_size - overlap):
                if label == "":
                    segments.append(text[j : j + chunk_size])
                else:
                    segments.append(
                        "[標題] " + label + " [/標題]. " + text[j : j + chunk_size]
                    )
                id.append(int(index))
        return pd.DataFrame({"id": id, "text": segments}), category
    elif category == "ground_truths":
        df = pd.DataFrame(data)
        return df
    else:
        raise ValueError("Invalid category")


def query_rewrite(query):
    n = [
        ("1", "一", "1月1日至3月31日"),
        ("2", "二", "4月1日至6月30日"),
        ("3", "三", "7月1日至9月30日"),
        ("4", "四", "10月1日至12月31日"),
    ]
    query_rewrite = query
    for season in n:
        if f"第{season[0]}季" in query or f"第{season[1]}季" in query:
            query_rewrite = query.replace(f"第{season[0]}季", season[2]).replace(
                f"第{season[1]}季", season[2]
            )
            break
    return query_rewrite


def jina_retrieve(
    insurance_path,
    finance_path,
    faq_path,
    ground_truth_path="/home/S113062628/project/AI-CUP-2024/dataset/preliminary/ground_truths_example.json",
    query_path="/home/S113062628/project/AI-CUP-2024/dataset/preliminary/questions_example.json",
    chunk_size_i=128,
    overlap_i=32,
    chunk_size_f=256,
    overlap_f=32,
    focus_on_source=True,
    summary=False,
    topk=1,
    name="",
):
    # Convert JSON to DataFrame with text and id columns for different categories
    df_insurance, _ = json_to_df(insurance_path, chunk_size_i, overlap_i, summary)
    df_finance, _ = json_to_df(finance_path, chunk_size_f, overlap_f, summary)
    df_faq = faq_to_df(faq_path)

    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)

    ground_truth = ground_truth["ground_truths"]

    with open(query_path, "r") as f:
        queries = json.load(f)

    queries = queries["questions"]

    insurance_queries = [query for query in queries if query["category"] == "insurance"]
    finance_queries = [query for query in queries if query["category"] == "finance"]
    faq_queries = [query for query in queries if query["category"] == "faq"]

    insurance_gt = [
        gt["retrieve"] for gt in ground_truth if gt["category"] == "insurance"
    ]
    finance_gt = [gt["retrieve"] for gt in ground_truth if gt["category"] == "finance"]
    faq_gt = [gt["retrieve"] for gt in ground_truth if gt["category"] == "faq"]

    categories = [
        (insurance_queries, insurance_gt, df_insurance, "Insurance"),
        (finance_queries, finance_gt, df_finance, "Finance"),
        (faq_queries, faq_gt, df_faq, "FAQ"),
    ]

    total_correct = 0
    total_queries = 0
    mismatches = []
    predictions = []

    for category_queries, category_gt, df, category_name in categories:
        correct = 0
        total = len(category_queries)

        # Iterate through each query with tqdm for progress visualization
        for i, query in enumerate(
            tqdm(category_queries, desc=f"Processing Queries for {category_name}")
        ):
            query_text = query["query"]
            query_text = query_rewrite(query_text)
            source_ids = set(query["source"])

            # Determine which passages to use (source only or all passages)
            if focus_on_source:
                relevant_passages = df[df["id"].isin(source_ids)]
            else:
                relevant_passages = df

            # If there are no relevant passages, skip the query
            if relevant_passages.empty:
                raise Exception(f"No relevant passages for query {i + 1}")

            # Create sentence pairs between the query and each relevant passage
            sentence_pairs = [
                [query_text, passage] for passage in relevant_passages["text"]
            ]

            # Compute similarity scores for each sentence pair
            scores = model.compute_score(sentence_pairs, max_length=1024)

            # Convert the list of scores to a tensor
            scores_tensor = torch.tensor(scores, device=device)

            # Find the passages with the highest scores up to the threshold
            topk_indices = torch.topk(scores_tensor, topk).indices.tolist()
            best_match_ids = relevant_passages.iloc[topk_indices]["id"].tolist()

            # Always store the top 5 scores, regardless of the value of topk
            top5_indices = torch.topk(scores_tensor, 5).indices.tolist()
            top5_match_ids = relevant_passages.iloc[top5_indices]["id"].tolist()

            # Compare the best match IDs with the ground truth
            if category_gt[i] in best_match_ids:
                correct += 1
            else:
                mismatches.append(
                    {
                        "qid": total_queries + i + 1,
                        "query": query_text,
                        "predicted": [
                            {
                                "id": int(match_id),
                                "score": float(scores_tensor[top5_indices[idx]].item()),
                                "passage": relevant_passages.iloc[top5_indices[idx]][
                                    "text"
                                ],
                            }
                            for idx, match_id in enumerate(top5_match_ids)
                        ],
                        "top1": int(best_match_ids[0]),
                        "ground_truth": int(category_gt[i]),
                    }
                )
            predictions.append(
                {
                    "qid": total_queries + i + 1,
                    "retrieve": int(best_match_ids[0]),
                }
            )

        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy for {category_name}: {accuracy * 100:.2f}%")

        total_correct += correct
        total_queries += total

    total_accuracy = total_correct / total_queries if total_queries > 0 else 0
    print(f"Total Accuracy: {total_accuracy * 100:.2f}%")

    with open(f"mismatch_{name}.json", "w") as f:
        json.dump(mismatches, f, ensure_ascii=False, indent=4)

    with open("pred_retrieve.json", "w") as f:
        json.dump({"answers": predictions}, f, ensure_ascii=False, indent=4)

    return total_accuracy


def jina_cross_encoder(
    insurance_path,
    finance_path,
    faq_path,
    ground_truth_path="/home/S113062628/project/AI-CUP-2024/dataset/preliminary/ground_truths_example.json",
    query_path="/home/S113062628/project/AI-CUP-2024/dataset/preliminary/questions_example.json",
    chunk_size_i=128,
    overlap_i=32,
    chunk_size_f=256,
    overlap_f=32,
    focus_on_source=True,
    summary=False,
    topk=1,
    name="",
):
    # Convert JSON to DataFrame with text and id columns for different categories
    df_insurance, _ = json_to_df(insurance_path, chunk_size_i, overlap_i, summary)
    df_finance, _ = json_to_df(finance_path, chunk_size_f, overlap_f, summary)
    df_faq = faq_to_df(faq_path)

    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)

    ground_truth = ground_truth["ground_truths"]

    with open(query_path, "r") as f:
        queries = json.load(f)

    queries = queries["questions"]

    insurance_queries = [query for query in queries if query["category"] == "insurance"]
    finance_queries = [query for query in queries if query["category"] == "finance"]
    faq_queries = [query for query in queries if query["category"] == "faq"]

    insurance_gt = [
        gt["retrieve"] for gt in ground_truth if gt["category"] == "insurance"
    ]
    finance_gt = [gt["retrieve"] for gt in ground_truth if gt["category"] == "finance"]
    faq_gt = [gt["retrieve"] for gt in ground_truth if gt["category"] == "faq"]

    categories = [
        (insurance_queries, insurance_gt, df_insurance, "Insurance"),
        (finance_queries, finance_gt, df_finance, "Finance"),
        (faq_queries, faq_gt, df_faq, "FAQ"),
    ]

    total_correct = 0
    total_queries = 0
    mismatches = []
    predictions = []

    for category_queries, category_gt, df, category_name in categories:
        correct = 0
        total = len(category_queries)

        # Iterate through each query with tqdm for progress visualization
        for i, query in enumerate(
            tqdm(category_queries, desc=f"Processing Queries for {category_name}")
        ):
            query_text = query["query"]
            source_ids = set(query["source"])

            # Determine which passages to use (source only or all passages)
            if focus_on_source:
                relevant_passages = df[df["id"].isin(source_ids)]
            else:
                relevant_passages = df

            # If there are no relevant passages, skip the query
            if relevant_passages.empty:
                continue

            # Create sentence pairs between the query and each relevant passage
            sentence_pairs = [
                [query_text, passage] for passage in relevant_passages["text"]
            ]

            # Compute similarity scores for each sentence pair
            scores = cross_encoder.predict(
                sentence_pairs, convert_to_tensor=True
            ).tolist()

            # Find the passage with the highest score
            best_match_idx = np.argmax(scores)
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
            predictions.append(
                {
                    "qid": total_queries + i + 1,
                    "retrieve": int(best_match_id),
                }
            )

        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy for {category_name}: {accuracy * 100:.2f}%")

        total_correct += correct
        total_queries += total

    total_accuracy = total_correct / total_queries if total_queries > 0 else 0
    print(f"Total Accuracy: {total_accuracy * 100:.2f}%")

    with open(f"mismatch_{name}.json", "w") as f:
        json.dump(mismatches, f, ensure_ascii=False, indent=4)

    with open("pred_retrieve.json", "w") as f:
        json.dump({"answers": predictions}, f, ensure_ascii=False, indent=4)

    return total_accuracy
