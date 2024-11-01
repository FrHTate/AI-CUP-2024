import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import re
from tqdm import tqdm


ngram_range = (1, 2)
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=ngram_range,
    token_pattern=r"(?u)\b\w+\b",
)

with open("reference/finance.json", "r", encoding="utf-8") as f:
    finance = json.load(f)
text = [" ".join(jieba.lcut(finance[idx]["text"])) for idx in range(len(finance))]

# 計算 TF-IDF
tfidf_vector = tfidf_vectorizer.fit_transform(text)
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = tfidf_vector.toarray()[0]
# 將詞彙與分數組成 DataFrame 並依分數排序
tfidf_df = pd.DataFrame({"word": feature_names, "score": tfidf_scores})
tfidf_df = tfidf_df.sort_values(by="score", ascending=False)
tfidf_df.to_csv("tfidf.csv", index=False)


# 移除分數較低的詞彙 <-- 可以想想移除的方法
removed_percentage = 0.2
removed_range = int(removed_percentage * len(tfidf_df))
low_tfidf_set = set(tfidf_df["word"].tail(removed_range))
low_tfidf_set = {word.replace(" ", "") for word in low_tfidf_set}
print(low_tfidf_set)


def remove_low_tfidf(text_list):
    filtered_text = []
    pattern = "|".join(low_tfidf_set)
    for text in tqdm(text_list, desc="Removing low tfidf words:"):
        filtered_text.append(re.sub(pattern, "", text["text"]))
    return filtered_text


if __name__ == "__main__":
    filtered_text = remove_low_tfidf(finance)

    ret_json = {
        "finance": [
            {
                "index": finance[idx]["index"],
                "label": filtered_text[idx][:50],
                "text": filtered_text[idx],
            }
            for idx in range(len(finance))
        ]
    }

    save_path = f"reference/filtered_finance_{ngram_range[1]}gram_r{int(removed_percentage*10)}%.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(ret_json, f, ensure_ascii=False, indent=4)
