import pandas as pd

def combine(file_path):
    first = pd.read_csv(f"{file_path}/first_partition.csv")
    second = pd.read_csv(f"{file_path}/second_partition.csv")
    third = pd.read_csv(f"{file_path}/third_partition.csv")
    fourth = pd.read_csv(f"{file_path}/fourth_partition.csv")
    fifth = pd.read_csv(f"{file_path}/fifth_partition.csv")
    sixth = pd.read_csv(f"{file_path}/sixth_partition.csv")
    combined = pd.concat([first, second, third, fourth, fifth, sixth])[["id","model_classification","reasoning"]]
    combined.to_csv(f"{file_path}/output.csv", index=False)

if __name__ == "__main__":
    combine("results/Teachers/Llama/meta-llama_Llama-3.3-70B-Instruct-Turbo-Free")
    combine("results/Teachers/Gemma/Gemma2_27b")