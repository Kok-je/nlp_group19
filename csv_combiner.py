import pandas as pd

# Inefficient but let's keep it for now since we used this for everything else
def preprocess(file_path):
    data = pd.read_json(path_or_buf=file_path, lines=True)
    ids = set()
    rows_to_be_dropped = []
    for i in range(len(data)):
        row = data.iloc[i]
        if row.unique_id in ids:
            rows_to_be_dropped.append(i)
        else:
            ids.add(row.unique_id)
    data = data.drop(rows_to_be_dropped)
    return data

def combine6(file_path):
    first = pd.read_csv(f"{file_path}/first_partition.csv")
    second = pd.read_csv(f"{file_path}/second_partition.csv")
    third = pd.read_csv(f"{file_path}/third_partition.csv")
    fourth = pd.read_csv(f"{file_path}/fourth_partition.csv")
    fifth = pd.read_csv(f"{file_path}/fifth_partition.csv")
    sixth = pd.read_csv(f"{file_path}/sixth_partition.csv")
    combined = pd.concat([first, second, third, fourth, fifth, sixth])[["id","model_classification","reasoning"]]
    combined.to_csv(f"{file_path}/output.csv", index=False)

def create_training_data(file_path,name = "training"):
    data = pd.read_csv(file_path)[["model_classification","reasoning"]].rename(columns={"classification":"model_classification","teachers_reasoning":"reasoning"}).reset_index(drop=True)
    og = preprocess("./data/train.jsonl")[["sectionName", "string", "unique_id"]].rename(columns={"sectionName":"sectionName"}).reset_index(drop=True)
    combined = og.merge(data,left_index = True,right_index = True, how="inner")

    assert combined.shape[0] == 8194
    combined = combined.map(lambda x : x if not isinstance(x,str) else x.replace("\n"," "),"ignore")
    combined.to_csv(f"Student_Training_Data/{name}.csv", index=False)
if __name__ == "__main__":
    combine6("results/Teachers/Llama/meta-llama_Llama-3.3-70B-Instruct-Turbo-Free")
    combine6("results/Teachers/Gemma/Gemma2_27b")
    combine6("results/Teachers/DeepSeek/R1")
    create_training_data("results/Teachers/deepseek-openai/deepseek_openai_combined.csv","Combined")