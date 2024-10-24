def build_catagory_csv(catagory):
    file_index = 0
    text = []
    index = []
    while file_path(catagory, file_index) is not None:
        try:
            text.append(read_pdf(file_path(catagory, file_index)))
            index.append(file_index)
            file_index += 1
        except:
            print(f"file {file_index} is broken")
            file_index += 1
    df = pd.DataFrame(text, index)
    df.to_csv(f"/home/S113062628/project/AI-CUP-2024/reference/{catagory}.csv")


build_catagory_csv("finance")
