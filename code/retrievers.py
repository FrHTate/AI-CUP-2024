import pandas as pd
import jieba
from ckiptagger import WS
from rank_bm25 import BM25Okapi

import torch
from transformers import AutoModelForSequenceClassification


def BM25_retrieve(
    query={"1by1", "full"}, corpus={"catbycat", "full"}, tokenizer="jieba", top_n=5
):
    source_corpus = corpus[corpus["id"].isin(source)]

    if tokenizer == "jieba":
        tokenized_query = list(jieba.cut_for_search(query))
        tokenized_corpus = [
            list(jieba.cut_for_search(doc)) for doc in source_corpus["text"]
        ]
    elif tokenizer == "ckiptagger":
        raise NotImplementedError("CKIP is not implemented yet")
    else:
        raise ValueError("Invalid tokenizer")

    bm25 = BM25Okapi(tokenized_corpus)

    doc_scores = bm25.get_scores(tokenized_query)
    top_n_idx = sorted(
        range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
    )[:top_n]

    return pd.DataFrame(
        {
            "id": [source_corpus.iloc[i]["id"] for i in top_n_idx],
            "score": [doc_scores[i] for i in top_n_idx],
        }
    )


# def jina_retrieve(query, source, corpus, top_n=5, chunk={}):
