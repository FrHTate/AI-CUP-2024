import json
import pandas as pd
import torch
import re
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


def chinese_to_arabic(chinese_numeral):
    chinese_numeral_map = {
        "零": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    result = 0

    if "十" in chinese_numeral:
        parts = chinese_numeral.split("十")
        if parts[0] == "":
            result += 10
        else:
            result += chinese_numeral_map[parts[0]] * 10
        if len(parts) > 1 and parts[1] != "":
            result += chinese_numeral_map[parts[1]]
    else:
        for char in chinese_numeral:
            result = result * 10 + chinese_numeral_map[char]

    return result


def convert_text_dates(text):
    # Convert 民國 (ROC year) to Gregorian year only for three-digit years,
    # ensuring not to convert already existing four-digit Gregorian years.
    text = re.sub(r"(?<!\d)(\d{3})年", lambda m: f"{int(m.group(1)) + 1911}年", text)

    # Convert fully Chinese representation of dates to Arabic numerals
    text = re.sub(
        r"([〇一二三四五六七八九十]+)年([〇一二三四五六七八九十]+)月([〇一二三四五六七八九十]+)日",
        lambda m: f"{chinese_to_arabic(m.group(1)) + 1911}年{chinese_to_arabic(m.group(2))}月{chinese_to_arabic(m.group(3))}日",
        text,
    )

    text = re.sub(
        r"([〇一二三四五六七八九十]+)年([〇一二三四五六七八九十]+)月",
        lambda m: f"{chinese_to_arabic(m.group(1)) + 1911}年{chinese_to_arabic(m.group(2))}月",
        text,
    )

    text = re.sub(
        r"([一二三四五六七八九十]+)月([〇一二三四五六七八九十]+)日",
        lambda m: f"{chinese_to_arabic(m.group(1))}月{chinese_to_arabic(m.group(2))}日",
        text,
    )

    return text


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
                    passage_rewrite(
                        text[j : j + chunk_size] + " [標題] " + label + " [/標題]."
                    )
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
                    segments.append(passage_rewrite(text[j : j + chunk_size]))
                else:
                    segments.append(
                        passage_rewrite(
                            "[標題] " + label + " [/標題]. " + text[j : j + chunk_size]
                        )
                    )
                id.append(int(index))
        return pd.DataFrame({"id": id, "text": segments}), category
    elif category == "ground_truths":
        df = pd.DataFrame(data)
        return df
    else:
        raise ValueError("Invalid category")


def query_rewrite(query):
    # years = re.findall(r"(\d{4})年", query)
    # if years:
    #     years = years[0]
    # else:
    #     years = ""
    n = [
        ("1", "一", f"Q1"),
        ("2", "二", f"Q2"),
        ("3", "三", f"Q3"),
        ("4", "四", f"Q4"),
    ]
    query_rewrite = query
    for season in n:
        if f"第{season[0]}季" in query or f"第{season[1]}季" in query:
            query_rewrite = query.replace(f"第{season[0]}季", season[2]).replace(
                f"第{season[1]}季", season[2]
            )

    query_rewrite = convert_text_dates(query_rewrite)

    company_names = {
        # "聯發科": "聯發科技股份有限公司",
        "台化": "台灣化學纖維股份有限公司",
        # "台達電": "台達電子工業股份有限公司",
        "台泥": "台灣水泥股份有限公司",
        # "華碩": "華碩電腦股份有限公司",
        # "瑞昱": "瑞昱半導體股份有限公司",
        # "長榮": "長榮海運股份有限公司",
        "聯電": "聯華電子股份有限公司",
        # "智邦": "智邦科技股份有限公司",
        # "和泰汽車": "和泰汽車股份有限公司",
        "中鋼": "中國鋼鐵股份有限公司",
        # "鴻海": "鴻海精密工業股份有限公司",
        # "亞德客": "亞德客國際集團及其子公司",
        # "統一企業": "統一企業股份有限公司",
        # "國巨": "國巨股份有限公司",
        # "研華": "研華股份有限公司",
        # "中華電信": "中華電信股份有限公司",
        # "光寶": "光寶科技股份有限公司",
        # "台積電": "台灣積體電路製造股份有限公司",
        # "台永電": "台灣永電股份有限公司",
        # "合作金庫": "合作金庫商業銀行股份有限公司",
    }

    for abbr, full_name in company_names.items():
        if abbr in query_rewrite and full_name not in query_rewrite:
            query_rewrite = query_rewrite.replace(abbr, f"{abbr}({full_name})")

    return query_rewrite


def passage_rewrite(passage):
    n = [
        ("1", "一", f"Q1"),
        ("2", "二", f"Q2"),
        ("3", "三", f"Q3"),
        ("4", "四", f"Q4"),
    ]
    passage_rewrite = passage
    for season in n:
        if f"第{season[0]}季" in passage or f"第{season[1]}季" in passage:
            passage_rewrite = passage_rewrite.replace(
                f"第{season[0]}季", season[2]
            ).replace(f"第{season[1]}季", season[2])

    # Fix the re.sub() calls by correctly using backreferences without additional quotes
    passage_rewrite = re.sub(
        r"(\d{4})年(\d{1,2})月(\d{1,2})日", r"\1/\2/\3", passage_rewrite
    )
    passage_rewrite = re.sub(r"(\d{4})年(\d{1,2})月", r"\1/\2", passage_rewrite)
    passage_rewrite = re.sub(r"(\d{1,2})月(\d{1,2})日", r"\1/\2", passage_rewrite)

    return passage_rewrite


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
                if query["qid"] in [53, 59, 62, 64, 68, 90, 93, 97, 99]:
                    mismatches.append(
                        {
                            "qid": total_queries + i + 1,
                            "query": query_text,
                            "predicted": [
                                {
                                    "id": int(match_id),
                                    "score": float(
                                        scores_tensor[top5_indices[idx]].item()
                                    ),
                                    "passage": relevant_passages.iloc[
                                        top5_indices[idx]
                                    ]["text"],
                                }
                                for idx, match_id in enumerate(top5_match_ids)
                            ],
                            "top1": int(best_match_ids[0]),
                            "ground_truth": int(category_gt[i]),
                        }
                    )
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
    ground_truth_path="/home/delic/Desktop/Code/AICup2024/AI-CUP-2024/dataset/preliminary/ground_truths_example.json",
    query_path="/home/delic/Desktop/Code/AICup2024/AI-CUP-2024/dataset/preliminary/questions_example.json",
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


# topk目前只能1, chunk_size目前只能1024
def jina_cross_encoder_doc(
    insurance_path,
    finance_path,
    faq_path,
    ground_truth_path="/home/S112062627/Code/AI-CUP-2024/dataset/preliminary/ground_truths_example.json",
    query_path="/home/S112062627/Code/AI-CUP-2024/dataset/preliminary/questions_example.json",
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
    """
    df_insurance, _ = json_to_df(insurance_path, chunk_size_i, overlap_i, summary)
    df_finance, _ = json_to_df(finance_path, chunk_size_f, overlap_f, summary)
    """
    df_insurance = pd.DataFrame(columns=["id", "text"])
    df_finance = pd.DataFrame(columns=["id", "text"])
    df_faq = faq_to_df(faq_path)

    id = []
    text = []
    with open(insurance_path, "r") as f:
        data = json.load(f)
    for doc in data["insurance"]:
        id.append(int(doc["index"]))
        text.append(doc["text"])
    df_insurance["id"] = id
    df_insurance["text"] = text

    id = []
    text = []
    with open(finance_path, "r") as f:
        data = json.load(f)
    for doc in data["finance"]:
        id.append(int(doc["index"]))
        text.append(doc["text"])
    df_finance["id"] = id
    df_finance["text"] = text

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

        if category_name == "Insurance":
            chunk_size = chunk_size_i
            overlap = overlap_i
        elif category_name == "Finance":
            chunk_size = chunk_size_f
            overlap = overlap_f
        elif category_name == "FAQ":
            chunk_size = 1024  # default value for jina reranker
            overlap = 80  # default value for jina reranker

        # Iterate through each query with tqdm for progress visualization
        for i, query in enumerate(
            tqdm(category_queries, desc=f"Processing Queries for {category_name}")
        ):
            query_text = query["query"]
            source_ids = set(query["source"])
            """
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
            """
            if focus_on_source:
                relevant_docs = df[df["id"].isin(source_ids)]
                # print(relevant_docs)
            else:
                relevant_docs = df

            if relevant_docs.empty:
                continue

            docs_candidates = relevant_docs["text"].tolist()

            result = model.rerank(
                query_text,
                docs_candidates,
                max_query_length=chunk_size // 2,  # at most half of max_query_length
                max_length=chunk_size,
                overlap=overlap,
                top_n=topk,
            )

            best_match_id = relevant_docs.iloc[result[0]["index"]]["id"]

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


if __name__ == "__main__":
    jina_cross_encoder_doc(
        insurance_path="/home/S112062627/Code/AI-CUP-2024/reference/insurance.json",
        finance_path="/home/S112062627/Code/AI-CUP-2024/reference/finance_artificial.json",
        faq_path="/home/S112062627/Code/AI-CUP-2024/reference/faq/pid_map_context.json",
        chunk_size_i=128,
        overlap_i=32,
        chunk_size_f=256,
        overlap_f=32,
        focus_on_source=True,
    )
