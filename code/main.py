# General
import pandas as pd
import os

# Our modules
import data_preprocesser as dp
import retrievers as rt

# 1-1. Set paths
path_to_folders = "/home/S112062627/Code/AI-CUP-2024"
paths = {
    "insurance": os.path.join(path_to_folders, "reference/insurance.json"),
    "finance": os.path.join(path_to_folders, "reference/finance_artificial.json"),
    "faq": os.path.join(path_to_folders, "reference/faq/pid_map_context.json"),
    "query": os.path.join(
        path_to_folders, "dataset/preliminary/questions_example.json"
    ),
    "output": os.path.join(path_to_folders, "dataset/preliminary/pred_retrieve.json"),
}

# 1-2. Set Hyperparameters
# If we don't want to chunk, just set "activate" to False
# Each value in top_k sould be either "full" or int
chunking = {
    "activate": True,
    "chunk_size": {"insurance": 128, "finance": 256, "jina_default": 1024},
    "overlap_size": {"insurance": 32, "finance": 32, "jina_default": 80},
}

top_k = {"BM25": 5, "jina": 5}

# 2. Load data
# Each item: {"qid": int, "source": list of str, "query": str, "category": str}
queries = dp.load_queries(paths["query"])

# Attributes: ["id": int, "text": str]
# df_insurance_full = dp.load_corpus_to_df("insurance", paths["insurance"])
# df_finance_full = dp.load_corpus_to_df("finance", paths["finance"])
# df_faq_full = dp.load_corpus_to_df("faq", paths["faq"])
df_insurance = dp.load_corpus_to_df(
    "insurance",
    paths["insurance"],
    chunking=chunking,
)
df_finance = dp.load_corpus_to_df(
    "finance",
    paths["finance"],
    chunking=chunking,
)
df_faq = dp.load_corpus_to_df("faq", paths["faq"])


# 3-1. Retreive by Jina-reranker
# Attributes: ["id": list of int, "text": list of str, "score": list of float]
df_jina_result = rt.jina_retrieve(
    queries=queries,
    insurance_corpus=df_insurance,
    finance_corpus=df_finance,
    faq_corpus=df_faq,
    chunking=chunking,
    top_k=top_k["jina"],
)

print(df_jina_result)
# 3-2. Retrieve by BM25
# Attributes: ["id": list of int, "text": list of str, "score": list of float]
df_BM25_result = rt.BM25_retrieve(
    queries=queries,
    insurance_corpus=df_insurance,
    finance_corpus=df_finance,
    faq_corpus=df_faq,
    tokenizer="jieba",
    top_k=top_k["BM25"],
)

print(df_BM25_result)


# 4. (Optional) Hybrid

# 5. Output the result


# 5. (Before Contest) Calculate Acc
ground_truths = dp.load_ground_truths(
    os.path.join(path_to_folders, "dataset/preliminary/ground_truths_example.json")
)

correct_jina = {"insurance": 0, "finance": 0, "faq": 0}
correct_BM25 = {"insurance": 0, "finance": 0, "faq": 0}
total = {"insurance": 0, "finance": 0, "faq": 0}

# 以下待改成function
for qid in range(len(queries)):
    if queries[qid]["category"] == "insurance":
        total["insurance"] += 1
        if df_jina_result.iloc[qid]["id"][0] == ground_truths[qid]["retrieve"]:
            correct_jina["insurance"] += 1
        if df_BM25_result.iloc[qid]["id"][0] == ground_truths[qid]["retrieve"]:
            correct_BM25["insurance"] += 1
    elif queries[qid]["category"] == "finance":
        total["finance"] += 1
        if df_jina_result.iloc[qid]["id"][0] == ground_truths[qid]["retrieve"]:
            correct_jina["finance"] += 1
        if df_BM25_result.iloc[qid]["id"][0] == ground_truths[qid]["retrieve"]:
            correct_BM25["finance"] += 1
    elif queries[qid]["category"] == "faq":
        total["faq"] += 1
        if df_jina_result.iloc[qid]["id"][0] == ground_truths[qid]["retrieve"]:
            correct_jina["faq"] += 1
        if df_BM25_result.iloc[qid]["id"][0] == ground_truths[qid]["retrieve"]:
            correct_BM25["faq"] += 1
print(
    f"Jina Insurance Acc: {correct_jina['insurance'] / total['insurance'] * 100:.2f}%"
)
print(f"Jina Finance Acc: {correct_jina['finance'] / total['finance'] * 100:.2f}%")
print(f"Jina FAQ Acc: {correct_jina['faq'] / total['faq'] * 100:.2f}%")
print(f"Jina Total Acc: {sum(correct_jina.values()) / sum(total.values()) * 100:.2f}%")

print(
    f"BM25 Insurance Acc: {correct_BM25['insurance'] / total['insurance'] * 100:.2f}%"
)
print(f"BM25 Finance Acc: {correct_BM25['finance'] / total['finance'] * 100:.2f}%")
print(f"BM25 FAQ Acc: {correct_BM25['faq'] / total['faq'] * 100:.2f}%")
print(f"BM25 Total Acc: {sum(correct_BM25.values()) / sum(total.values()) * 100:.2f}%")
