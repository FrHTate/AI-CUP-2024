import torch
from transformers import BertTokenizer, BertModel
import faiss
import numpy as np
from tqdm import tqdm
import os
import json
import argparse
import jieba  # 用於中文文本分詞
import pdfplumber  # 用於從PDF文件中提取文字的工具
from bm25_retrieve import BM25_retrieve

tokenizer = BertTokenizer.from_pretrained("ckiplab/bert-base-chinese")
model = BertModel.from_pretrained("ckiplab/bert-base-chinese")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using GPU")
else:
    print("Using CPU")
model = model.to(device)


def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {
        int(file.replace(".pdf", "")): read_pdf(os.path.join(source_path, file))
        for file in tqdm(masked_file_ls, desc="Loading data")
    }  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    return corpus_dict


# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0] : page_infos[1]] if page_infos else pdf.pages
    pdf_text = ""
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萃取出的文本


def encode_corpus(corpus):
    encoded_corpus = []
    for document in tqdm(corpus, desc="Encoding corpus"):
        inputs = tokenizer(
            document,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = (
            outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        )  # [CLS] token embedding
        encoded_corpus.append(cls_embeddings)
    return np.array(encoded_corpus)


# Encode a query using BERT
def encode_query(query):
    inputs = tokenizer(
        query, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return (
        outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    )  # [CLS] token embedding


# Perform retrieval with FAISS
def dense_retrieve(query, source, corpus_vectors):
    query_vector = encode_query(query)
    index = faiss.IndexFlatL2(corpus_vectors.shape[1])  # Create FAISS index
    source_vectors = np.array([corpus_vectors[idx] for idx in source])
    index.add(source_vectors)
    D, I = index.search(query_vector.reshape(1, -1), k=1)
    return source[I[0][0]]


if __name__ == "__main__":

    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description="Process some paths and files.")
    parser.add_argument(
        "--question_path",
        type=str,
        required=False,
        help="讀取發布題目路徑",
        default="/home/S113062628/project/AI-CUP-2024/dataset/preliminary/questions_example.json",
    )  # 問題文件的路徑
    parser.add_argument(
        "--source_path",
        type=str,
        required=False,
        help="讀取參考資料路徑",
        default="/home/S113062628/project/AI-CUP-2024/reference",
    )  # 參考資料的路徑
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        help="輸出符合參賽格式的答案路徑",
        default="/home/S113062628/project/AI-CUP-2024/dataset/preliminary/answers.json",
    )  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    with open(args.question_path, "rb") as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    source_path_insurance = os.path.join(args.source_path, "insurance")

    source_path_finance = os.path.join(args.source_path, "finance")  # 設定參考資料路徑

    corpus_dict_insurance = load_data(source_path_insurance)
    corpus_vectors_insurance = encode_corpus(
        corpus_dict_insurance.values()
    )  # Encode corpus to dense vectors

    corpus_dict_finance = load_data(source_path_finance)
    corpus_vectors_finance = encode_corpus(
        corpus_dict_finance.values()
    )  # Encode corpus to dense vectors

    with open(os.path.join(args.source_path, "faq/pid_map_content.json"), "rb") as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {
            int(key): value for key, value in key_to_source_dict.items()
        }

    for q_dict in qs_ref["questions"]:
        if q_dict["category"] == "finance":
            retrieved_idx = dense_retrieve(
                q_dict["query"], q_dict["source"], corpus_vectors_finance
            )
            answer_dict["answers"].append(
                {"qid": q_dict["qid"], "retrieve": int(retrieved_idx)}
            )

        elif q_dict["category"] == "insurance":
            retrieved_idx = dense_retrieve(
                q_dict["query"], q_dict["source"], corpus_vectors_insurance
            )
            answer_dict["answers"].append(
                {"qid": q_dict["qid"], "retrieve": int(retrieved_idx)}
            )
        elif q_dict["category"] == "faq":
            corpus_dict_faq = {
                key: str(value)
                for key, value in key_to_source_dict.items()
                if key in q_dict["source"]
            }
            retrieved = BM25_retrieve(
                q_dict["query"], q_dict["source"], corpus_dict_faq
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤
        # Handle the FAQ similarly by encoding its vectors and using dense_retrieve

    with open(args.output_path, "w", encoding="utf8") as f:
        json.dump(
            answer_dict, f, ensure_ascii=False, indent=4
        )  # 儲存檔案，確保格式和非ASCII字符
