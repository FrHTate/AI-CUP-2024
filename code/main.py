from helper_process import (
    passage_embedding,
    query_embedding,
    retriever,
    accuracy_of_category,
)

finance_path = "/home/S113062628/project/AI-CUP-2024/reference/finance.json"
# isinstance_path = "/home/S113062628/project/AI-CUP-2024/reference/filtered_insurance.json"
query_path = (
    "/home/S113062628/project/AI-CUP-2024/dataset/preliminary/questions_example.json"
)
# passage_embedding(finance_path, truncate_size=256, overlap=32)
# query_embedding(query_path)

path = "/home/S113062628/project/AI-CUP-2024/embeddings"
finance = path + "/finance_embeddings.csv"
query = path + "/query.csv"

answer = retriever(query, finance)

ground_truth = [
    392,
    428,
    83,
    186,
    162,
    116,
    107,
    78,
    62,
    472,
    7,
    526,
    526,
    526,
    536,
    54,
    606,
    184,
    315,
    292,
    36,
    614,
    224,
    359,
    4,
    147,
    171,
    298,
    578,
    327,
    10,
    434,
    434,
    442,
    319,
    148,
    148,
    353,
    29,
    591,
    165,
    325,
    325,
    440,
    537,
    66,
    620,
    8,
    482,
    78,
    162,
    918,
    351,
    612,
    166,
    171,
    668,
    209,
    632,
    726,
    900,
    591,
    306,
    124,
    255,
    192,
    1021,
    942,
    981,
    490,
    920,
    204,
    235,
    569,
    71,
    211,
    843,
    119,
    65,
    213,
    671,
    831,
    155,
    745,
    891,
    189,
    701,
    256,
    793,
    710,
    22,
    55,
    660,
    699,
    307,
    435,
    282,
    692,
    372,
    273,
    558,
    104,
    63,
    15,
    294,
    224,
    540,
    105,
    283,
    611,
    76,
    403,
    509,
    279,
    54,
    4,
    92,
    554,
    610,
    20,
    414,
    415,
    527,
    14,
    604,
    1,
    365,
    436,
    283,
    5,
    600,
    334,
    243,
    328,
    28,
    504,
    359,
    339,
    194,
    141,
    410,
    173,
    144,
    225,
    564,
    0,
    87,
    242,
    441,
    221,
]

print(accuracy_of_category(ground_truth[], answer))
