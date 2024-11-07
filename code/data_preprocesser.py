import json
import pandas as pd


# chunking = (Yes/No, chunk_size)
def load_corpus_to_df(
    category,
    source_path,
    chunk={"activate": False, "chunk_size": None, "overlap_size": None},
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
        if chunk["activate"]:
            for doc in data[category]:
                if summary:
                    id_list.append(int(doc["index"]))
                    text_list.append(doc["summary"])
                for i in range(
                    0,
                    len(doc["text"]) - chunk["chunk_size"] + 1,
                    chunk["chunk_size"] - chunk["overlap_size"],
                ):
                    id_list.append(int(doc["index"]))
                    text_list.append(
                        doc["text"][i : i + chunk["chunk_size"]]
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
        if chunk["activate"]:
            for doc in data[category]:
                if summary:
                    id_list.append(int(doc["index"]))
                    text_list.append(doc["summary"])
                for i in range(
                    0,
                    len(doc["text"]) - chunk["chunk_size"] + 1,
                    chunk["chunk_size"] - chunk["overlap_size"],
                ):
                    id_list.append(int(doc["index"]))
                    if doc["label"] == "":
                        text_list.append(doc["text"][i : i + chunk["chunk_size"]])
                    else:
                        text_list.append(
                            "[標題] "
                            + doc["label"]
                            + " [/標題]. "
                            + doc["text"][i : i + chunk["chunk_size"]]
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
    # If there is no rewrite, return the original query
    query_rewrite = query
    seasons = [
        ("1", "一", "1月1日至3月31日"),
        ("2", "二", "4月1日至6月30日"),
        ("3", "三", "7月1日至9月30日"),
        ("4", "四", "10月1日至12月31日"),
    ]

    for season in seasons:
        if (f"第{season[0]}季" in query) or (f"第{season[1]}季" in query):
            query_rewrite = query.replace(f"第{season[0]}季", season[2]).replace(
                f"第{season[1]}季", season[2]
            )
            break
    return query_rewrite


def passage_rewrite(passage):
    # If there is no rewrite, return the original passage
    passage_rewrite = passage
    seasons = [
        ("1", "一", "1月1日至3月31日"),
        ("2", "二", "4月1日至6月30日"),
        ("3", "三", "7月1日至9月30日"),
        ("4", "四", "10月1日至12月31日"),
    ]

    for season in seasons:
        if f"第{season[0]}季" in passage or f"第{season[1]}季" in passage:
            passage_rewrite = passage.replace(f"第{season[0]}季", season[2]).replace(
                f"第{season[1]}季", season[2]
            )
            break
    return passage_rewrite
