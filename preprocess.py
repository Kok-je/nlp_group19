import pandas as pd    

def preprocess(file_path):
    data = pd.read_json(path_or_buf=file_path, lines=True)
    
    print(data.head())

preprocess("./data/train.jsonl")