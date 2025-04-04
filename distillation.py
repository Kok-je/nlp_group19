# %pip install -q transformers datasets accelerate peft bitsandbytes sentence-transformers

import nbformat
print(nbformat.__version__)

import pandas as pd
from datasets import Dataset
import torch
import time
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)
from dotenv import load_dotenv
load_dotenv()

def load_dataset_with_context(partition_path: str) -> Dataset:
    df = pd.read_csv(partition_path)
    return Dataset.from_pandas(df[["sectionName", "string", "id", "model_classification", "reasoning"]])

train_dataset = load_dataset_with_context("./merged_dataset.csv")

# %run student_eval.ipynb

from student_eval import call_pipe

hf_token = os.getenv('HUGGINGFACE_API_KEY')
model_id = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    token=hf_token,
    attn_implementation="eager"
)

def collate_fn(examples):
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    metadata = {
        "sectionName": [],
        "string": [],
        "teacher_reasoning": []
    }
    
    for example in examples:
        prompt = f"Classify this citation:\nSection: {example['sectionName']}\nText: {example['string']}\nClassification:"
        tokenized = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        batch["input_ids"].append(tokenized["input_ids"])
        batch["attention_mask"].append(tokenized["attention_mask"])
        batch["labels"].append(tokenized["input_ids"].clone())
        
        metadata["sectionName"].append(example["sectionName"])
        metadata["string"].append(example["string"])
        metadata["teacher_reasoning"].append(example["reasoning"])
    
    batch["input_ids"] = torch.cat(batch["input_ids"], dim=0)
    batch["attention_mask"] = torch.cat(batch["attention_mask"], dim=0)
    batch["labels"] = torch.cat(batch["labels"], dim=0)
    batch.update(metadata)
    
    return batch

class CosineSimilarityDistiller(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.student_pipe = None
        self.step_counter = 0
        
    def setup_pipeline(self):
        if not self.student_pipe:
            self.student_pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.model.device,
                max_new_tokens=2048
            )
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        print(f"\n[DEBUG] compute_loss() called at step {self.step_counter}")
        # Standard language modeling loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Cosine similarity alignment every 5 steps
        if self.step_counter % 5 == 0:
            print("[DEBUG] Running cosine similarity alignment...")
            self.setup_pipeline()
            
            # Generate student outputs
            student_reasonings = []
            for section, text in zip(inputs["sectionName"], inputs["string"]):
                _, reasoning = call_pipe(self.student_pipe, section, text)
                student_reasonings.append(reasoning)
                
            print(f"[DEBUG] Generated {len(student_reasonings)} student reasonings")
            # Get embeddings
            teacher_embeds = self.get_embeddings(inputs["teacher_reasoning"])
            student_embeds = self.get_embeddings(student_reasonings)
            
            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(teacher_embeds, student_embeds)
            similarity_loss = 1 - cos_sim.mean()
            
            # Combine losses
            total_loss = loss + (0.5 * similarity_loss)
            
            if self.step_counter % 10 == 0:
                print(f"Step {self.step_counter} - "
                      f"LM Loss: {loss:.4f} | "
                      f"Similarity Loss: {similarity_loss:.4f} | "
                      f"Total Loss: {total_loss:.4f}")
        else:
            total_loss = loss
            print(f"[DEBUG] Language model loss computed: {loss.item():.4f}")
        
        self.step_counter += 1
        return (total_loss, outputs) if return_outputs else total_loss
    
    def get_embeddings(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-1][:, 0, :]  # CLS token embedding

training_args = TrainingArguments(
    output_dir="cosine-distilled",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    eval_steps=100,
    remove_unused_columns=False
)

trainer = CosineSimilarityDistiller(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    tokenizer=tokenizer
)

trainer.train()

