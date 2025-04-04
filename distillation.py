import pandas as pd
from datasets import Dataset
import torch
import time
import os
import json
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on device: {device}")

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
        ).to(device)
        
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
    per_device_train_batch_size=2,
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

