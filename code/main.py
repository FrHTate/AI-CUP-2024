# Modules from pip
import pandas as pd
import os

# Our modules
from helper_jina import jina_retrieve, jina_cross_encoder

# Define paths
path_to_folders = "/home/S112062627/Code/AI-CUP-2024"
paths = {
    "insurance": os.path.join(path_to_folders, "reference/insurance.json"),
    "finance": os.path.join(path_to_folders, "reference/finance_artificial.json"),
    "faq": os.path.join(path_to_folders, "reference/faq/pid_map_context.json"),
    "question": os.path.join(
        path_to_folders, "dataset/preliminary/questions_example.json"
    ),
    "ground_truth": os.path.join(
        path_to_folders, "dataset/preliminary/ground_truth_example.json"
    ),
}

# Load data


# Retrieve top-5 by BM25

# Retreive top-5 by Jina-reranker (rerank provided sources)

# Reranking
