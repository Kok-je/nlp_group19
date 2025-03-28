import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

def initialize_model(model_name: str):
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    
    if not api_key:
        raise ValueError("No API key found. Please set the HUGGINGFACE_API_KEY environment variable.")
    
    client = InferenceClient(provider="hf-inference", api_key=api_key)
    return client, model_name

if __name__ == "__main__":
    MODEL_NAME = "dslim/bert-base-NER"

    client, model_name = initialize_model(MODEL_NAME)

    result = client.token_classification(
        text="My name is Shawn Kok but you can call me Captain America.",
        model=model_name
    )

    print("Model Output:", result)