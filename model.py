import json
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import pipeline

load_dotenv()

def initialize_model(model_name: str):
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    
    if not api_key:
        raise ValueError("No API key found. Please set the HUGGINGFACE_API_KEY environment variable.")
    
    client = InferenceClient(provider="hf-inference", api_key=api_key)
    return client, model_name

def call_llm(client, model_name, section_name, content):
    # messages = [{"role": "system", "content": "You are an AI assistant who will assess"
    # " the text, and the corresponding section name given in the input, classify the text as one of "
    # "[result, background, method] and provide the resoning for "
    # "the classification. Format of input would look as such, section name: name_of_section1, text: text1\nsection name: name_of_section2, text: text2."
    # "There will be multiple inputs to classify and reason about, and the output should contain all answers to all inputs, separated by a new line"
    # "Format of the output for each classification should contain only 2 things, the classification value,"
    # " and the resoning, separated by a new line character. "
    # "Example output would look like: background\nThis is the reasoning\nresult\nThis is the second reasoning"
    # }, 
    # {"role": "user", "content": content}]
    messages = [{"role": "system", "content": "You are an AI assistant who will assess"
    " the text, and the corresponding section name given in the input, classify the text as one of "
    "[result, background, method] and provide the resoning for "
    "the classification. Format of input would look as such, section name: name_of_section1, text: text1"
    "Format of the output for each classification should contain only 2 things, the classification value,"
    " and the resoning, separated by a new line character. "
    "Example output would look like: \"background\nThis is the reasoning\". The output must not contain any additional new line characters."
    }, 
    {"role": "user", "content": f"section name: {section_name}, text: {content}"}]
    response = client.chat_completion(model="meta-llama/Meta-Llama-3-8B-Instruct", messages=messages, max_tokens=2048)
    print(response)
    return response.choices[0].message.content.split("\n")


if __name__ == "__main__":
    MODEL_NAME = "dslim/bert-base-NER"

    client, model_name = initialize_model(MODEL_NAME)

    result = call_llm(client, "", "Introduction", "However, how frataxin interacts with the Fe-S cluster biosynthesis components remains unclear as direct"
    "one-to-one interactions with each component were reported (IscS [12,22], IscU/Isu1 [6,11,16] or ISD11/Isd11 [14,15]).")

    print(result)
