# Modules from pip
import pandas as pd
import os
import gdown

# Our modules
import data_preprocesser as dp
import retrievers as rt
from helper_jina import jina_retrieve

# from helper_jina import jina_retrieve, jina_cross_encoder

# Download utils for tokenization

# Define paths
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

# Hyperparameters
chunk_insurance = {"activate": True, "chunk_size": 128, "overlap_size": 32}
chunk_finance = {"activate": True, "chunk_size": 256, "overlap_size": 32}

# Load (and preprocessed) data
# Attributes: ["id": int, "text": str]
# df_insurance_full = dp.load_corpus_to_df("insurance", paths["insurance"])
# df_finance_full = dp.load_corpus_to_df("finance", paths["finance"])
# df_faq_full = dp.load_corpus_to_df("faq", paths["faq"])
df_insurance = dp.load_corpus_to_df(
    "insurance",
    paths["insurance"],
    chunk=chunk_insurance,
)
df_finance = dp.load_corpus_to_df(
    "finance",
    paths["finance"],
    chunk=chunk_finance,
)
df_faq = dp.load_corpus_to_df("faq", paths["faq"])

# Each item: {"qid": int, "source": list of str, "query": str, "category": str}
queries = dp.load_queries(paths["query"])

# Retreive top-5 by Jina-reranker (rerank provided sources)

# Retrieve top-5 by BM25

# Hybrid
