def load_corpus_to_df(source_path):
    # attributes in df = ["id", "text"]
    id = []
    text = []
    
    with open(source_path, "r") as f:
        data = json.load(f)

    if 