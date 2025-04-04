import pandas as pd
# Use a pipeline as a high-level helper
from datasets import Dataset
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os


from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv('HUGGINGFACE_API_KEY')
model_id = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    token=hf_token
)

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="google/gemma-3-1b-it", max_new_tokens=2048)
pipe(messages)

def call_pipe(pipe, section_name, content):
    messages = [{"role": "system", "content": 
                 "You are an AI assistant who will assess whether a given text is used as a background, result or method section in a scientific paper."
                 "You will be given a section name and the text, and you will need to classify the text as one of [background, result, method]."
                 "You will also need to provide a reasoning for your classification."
                 "The output should strictly be a json with the following two keys: classification, reasoning."
                 "Example output would look like: {\"classification\": \"result\", \"reasoning\": \"This is the reasoning\"}"
    }, 
    {"role": "user", "content": f"section name: {section_name}, text: {content}"}]
    response = pipe(messages)[0]
    try:
        if response["generated_text"][2]["content"]:
            print(response["generated_text"][2]["content"])
            content = response["generated_text"][2]["content"]
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
            answers = json.loads(json_str)
            classification = answers["classification"]
            reasoning = answers["reasoning"]
            return classification, reasoning
        else:
            return "Invalid", "No reasoning provided"
    except Exception as e:
        print(e)
        return "Invalid", "No reasoning provided"


result = call_pipe(pipe, "Introduction", "However, how frataxin interacts with the Fe-S cluster biosynthesis components remains unclear as direct one-to-one interactions with each component were reported (IscS [12,22], IscU/Isu1 [6,11,16] or ISD11/Isd11 [14,15]).")

print(result)


# def preprocess(file_path):
#     data = pd.read_json(path_or_buf=file_path, lines=True)
#     ids = set()
#     rows_to_be_dropped = []
#     for i in range(len(data)):
#         row = data.iloc[i]
#         if row.unique_id in ids:
#             rows_to_be_dropped.append(i)
#         else:
#             ids.add(row.unique_id)
#     data = data.drop(rows_to_be_dropped)
#     return data

# data = preprocess("./data/train.jsonl")

# labels = []
# reasonings = []
# raw_output = []
# ids = []

# for i in range(len(data)):
#     current_data = data.iloc[i]
#     ids.append(current_data.unique_id)
#     classification, reasoning = call_pipe(pipe, current_data.sectionName, current_data.string)
#     raw_output.append(classification)
#     labels.append(classification)
#     reasonings.append(reasoning)

# df = pd.DataFrame(zip(ids, labels, reasonings), columns=["id", "model_classification", "reasoning"])
# df.to_csv("llama-1b.csv")

