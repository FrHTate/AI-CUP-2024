import json
import pandas as pd
import re


def load_corpus_to_df(
    category,
    source_path,
    chunking={"activate": False, "chunk_size": None, "overlap_size": None},
    summary=False,
):
    # attributes in df = ["id", "text"]
    id_list = []
    text_list = []  # if chunking, text == segment; else, text == full text
    with open(source_path, "r") as f:
        data = json.load(f)

    if category == "faq":
        for id, qa_list in data.items():
            for qa in qa_list:
                for answer in qa["answers"]:
                    id_list.append(int(id))
                    text_list.append(qa["question"] + answer)
        return pd.DataFrame({"id": id_list, "text": text_list})

    elif category == "insurance":
        if chunking["activate"]:
            chunk_size = chunking["chunk_size"][category]
            overlap_size = chunking["overlap_size"][category]
            for doc in data[category]:
                if summary:
                    id_list.append(int(doc["index"]))
                    text_list.append(doc["summary"])
                for i in range(
                    0,
                    len(doc["text"]) - chunk_size + 1,
                    chunk_size - overlap_size,
                ):
                    id_list.append(int(doc["index"]))
                    text_list.append(
                        doc["text"][i : i + chunk_size]
                        + " [標題] "
                        + doc["label"]
                        + " [/標題]."
                    )
            return pd.DataFrame({"id": id_list, "text": text_list})

        else:
            for doc in data[category]:
                id_list.append(int(doc["index"]))
                text_list.append(doc["text"])
            return pd.DataFrame({"id": id_list, "text": text_list})

    elif category == "finance":
        if chunking["activate"]:
            chunk_size = chunking["chunk_size"][category]
            overlap_size = chunking["overlap_size"][category]
            for doc in data[category]:
                if summary:
                    id_list.append(int(doc["index"]))
                    text_list.append(doc["summary"])
                for i in range(
                    0,
                    len(doc["text"]) - chunk_size + 1,
                    chunk_size - overlap_size,
                ):
                    id_list.append(int(doc["index"]))
                    if doc["label"] == "":
                        text_list.append(doc["text"][i : i + chunk_size])
                    else:
                        text_list.append(
                            "[標題] "
                            + doc["label"]
                            + " [/標題]. "
                            + doc["text"][i : i + chunk_size]
                        )
            return pd.DataFrame({"id": id_list, "text": text_list})

        else:
            for doc in data[category]:
                id_list.append(int(doc["index"]))
                text_list.append(doc["text"])
            return pd.DataFrame({"id": id_list, "text": text_list})

    else:
        raise ValueError(
            "Invalid category, please choose from 'faq', 'insurance', 'finance'"
        )


def load_queries(source_path):
    with open(source_path, "r") as f:
        data = json.load(f)
    return data["questions"]


# Use for preparing stage, contest stage won't provide ground truth
def load_ground_truths(source_path):
    with open(source_path, "r") as f:
        data = json.load(f)
    return data["ground_truths"]


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
