import os
import time
import argparse
import torch
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")

def load_partition(path: str) -> Dataset:
    df = pd.read_csv(path)
    return Dataset.from_pandas(df)

dataset = load_partition("./merged_dataset.csv")

# ====== Tokenizer & Model Setup ======
model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    trust_remote_code=True,
    # torch_dtype=torch.float16,
    # quantization_config=bnb_config,
)

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# ====== Format data ======
def format_for_distillation(examples):
    prompts, responses, reasonings = [], [], []
    for text, reasoning, classification in zip(examples["string"], examples["reasoning"], examples["model_classification"]):
        prompt = (f"<instruction>Classify the following scientific text as one of [background, method, result].\n\n"
                  f"Text: {text}\n"
                  f"Provide your classification and reasoning in JSON format.</instruction>")
        response = f'<response>{{"classification": "{classification}", "reasoning": "{reasoning}"}}'
        prompts.append(prompt + response)
        reasonings.append(reasoning)

    tokenized = tokenizer(prompts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    labels = tokenized["input_ids"].clone()

    # Mask out instruction part
    for i, input_ids in enumerate(tokenized["input_ids"]):
        response_ids = tokenizer("<response>", add_special_tokens=False)["input_ids"]
        for j in range(len(input_ids) - len(response_ids)):
            if input_ids[j:j+len(response_ids)].tolist() == response_ids:
                labels[i, :j+len(response_ids)] = -100
                break

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
        "student_reasoning": reasonings  # Keep for Phase 2
    }

tokenized_dataset = dataset.map(format_for_distillation, batched=True, remove_columns=["id"])

# ====== Training Args ======
training_args = TrainingArguments(
    output_dir="llama-student-phase1-debug",
    num_train_epochs=3, # increased from 2
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5, # decreased from 1e-4
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
    max_grad_norm=1.0,
    report_to="none",
    logging_dir='./llama-student-phase1-logs', 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
model.save_pretrained("llama-student-phase1-debug")
tokenizer.save_pretrained("llama-student-phase1-debug")