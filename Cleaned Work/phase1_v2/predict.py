import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, classification_report
import os
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")

model_path = 'distilled_t5_on_8194_samples'
# Configuration
# model_path = f"{new_trained_model_name}" ## TODO: Note that this model is trained only on a 1000 samples! Because the paper says 25% of full training ata was alr good enough, so i wanted to just test with a smaller number of samples first.
test_data_path = "data/test.jsonl"
device = torch.device('cuda')

# Load the distilled model
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# After loading tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def load_test_data(file_path):
    """Load and parse test data"""
    test_data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            test_data.append({
                "section": entry["sectionName"],
                "text": entry["string"],
                "true_label": entry["label"]
            })
    return test_data

def preprocess_input(section, text):
    """Format input with task prefix"""
    input_text = f"[label] Section: {section}\nText: {text} \nLabel (either background, method or result):" ## TODO: NOTE THAT THIS IS KEYyyyy
    return tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

def predict_label(model, inputs):
    """Generate label prediction"""
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            # For deterministic results (default):
            do_sample=False,  # Disables sampling
            num_beams=3,     # Beam search works better for Seq2Seq
            early_stopping=False,
            # Remove temperature parameter when do_sample=False
            decoder_start_token_id=tokenizer.pad_token_id, #critical for T5
            pad_token_id=tokenizer.pad_token_id,
            # forced_bos_token_id=tokenizer.convert_tokens_to_ids("method"),
            # eos_token_id=tokenizer.eos_token_id,
        )

    # Debug raw outputs
    print("Raw output IDs:", outputs[0])
    # print("Decoded output:", tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def clean_prediction(raw_prediction):
    """Extract label from model output"""
    # Split on "Label:" and take the first word after it
    print(f"Raw: {raw_prediction}")
    # parts = raw_prediction.split("Label:")
    if len(raw_prediction) > 1:
        prediction = raw_prediction.strip().split()[0].lower()
        # Map to valid labels
        valid_labels = {"background", "method", "result"}
        print(f"Prediction: {prediction}")
        return prediction if prediction in valid_labels else "unknown"
    return "unknown"

# Load test data
test_data = load_test_data(test_data_path)

# Run predictions
true_labels = []
pred_labels = []

for example in test_data:
    # Preprocess input
    inputs = preprocess_input(example["section"], example["text"])
    
    # Get prediction
    raw_pred = predict_label(model, inputs)
    cleaned_label = clean_prediction(raw_pred)
    
    # Store results
    true_labels.append(example["true_label"])
    pred_labels.append(cleaned_label)
    
    # Print example (optional)
    print(f"Section: {example['section']}")
    print(f"Text: {example['text'][:100]}...")
    print(f"True: {example['true_label']} | Pred: {cleaned_label}")
    print("-" * 80)

# Calculate accuracy
accuracy = accuracy_score(true_labels, pred_labels)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Save results
with open("predictions_t5_trained_full.csv", "w") as f:
    f.write("true_label,predicted_label\n")
    for true, pred in zip(true_labels, pred_labels):
        f.write(f"{true},{pred}\n")
