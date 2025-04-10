import os
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")

# ====== Tokenizer & Model Setup ======
model_id = "google-bert/bert-base-uncased" #"google/gemma-3-1b-it"

# tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("t5-base", token=hf_token, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base", 
                                            # load_in_8bit=True,
                                            device_map="auto",
                                            #  quantization_config=BitsAndBytesConfig(
                                            #      load_in_8bit=True,
                                            #      llm_int8_threshold=6.0,
                                            #      llm_int8_enable_fp32_cpu_offload=True,
                                            #  ),
                                            #  trust_remote_code=True,
                                             )

#Ensure tokenizer has special tokens:
tokenizer.add_special_tokens({
    'additional_special_tokens': ['[label]', '[rationale]']
})
model.resize_token_embeddings(len(tokenizer))

# ====== Load dataset ======
def load_partition(path: str) -> Dataset:
    df = pd.read_csv(path) #.head(10)
    return Dataset.from_pandas(df)

dataset = load_partition("/Users/evedaktyl/Documents/y3s2/4248/proj/nlp_group19/Student_Training_Data/The_King.csv") 
print(f"Loaded {len(dataset)} samples from dataset.") 

def add_special_tokens_if_missing(tokenizer):
    # Add task-specific tokens if not present
    special_tokens = []
    if "[label]" not in tokenizer.get_vocab():
        special_tokens.append("[label]")
    if "[rationale]" not in tokenizer.get_vocab():
        special_tokens.append("[rationale]")
    
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer

# Update tokenizer with special tokens
tokenizer = add_special_tokens_if_missing(tokenizer)

def tokenize_function(examples):
    # Create base text inputs
    base_texts = [
        f"Section Name: {sn}\nText: {txt}" 
        for sn, txt in zip(examples["sectionName"], examples["string"])
    ]

    # print(f"Base texts: {base_texts}")

    # Create task-specific inputs
    label_inputs = [f"[label] {text} \nLabel (either background, method or result):" for text in base_texts]
    rationale_inputs = [f"[rationale] {text} \nRationale:" for text in base_texts]
    print(f"Label inputs: {label_inputs}")
    print(f"Rationale inputs: {rationale_inputs}")

    # Tokenize base inputs (for potential shared encoder)
    base_encoded = tokenizer(
        base_texts,
        padding="max_length",
        truncation=True,
        max_length=256,  # Reserve space for prefixes
        return_tensors="pt"
    )

    # Tokenize label task inputs and targets
    label_encoded = tokenizer(
        label_inputs,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Tokenize label targets (text labels, not indices)
    label_targets = tokenizer(
        examples["model_classification"],
        padding="max_length",
        truncation=True, 
        max_length=32,  # Short length for class labels
        return_tensors="pt"
    )

    # Tokenize rationale task inputs
    rationale_encoded = tokenizer(
        rationale_inputs,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Tokenize rationale targets
    rationale_targets = tokenizer(
        examples["reasoning"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    return {
        # Base inputs (shared between tasks)
        "base_input_ids": base_encoded.input_ids,
        "base_attention_mask": base_encoded.attention_mask,

        # Label prediction task
        "label_input_ids": label_encoded.input_ids,
        "label_attention_mask": label_encoded.attention_mask,
        "label_target_ids": label_targets.input_ids,

        # Rationale generation task
        "rationale_input_ids": rationale_encoded.input_ids,
        "rationale_attention_mask": rationale_encoded.attention_mask,
        "rationale_target_ids": rationale_targets.input_ids,
    }

# Apply tokenization
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32,
    remove_columns=dataset.column_names  # Remove original columns
)

# Set format for PyTorch
tokenized_dataset.set_format(type="torch", columns=[
    "base_input_ids",
    "base_attention_mask",
    "label_input_ids",
    "label_attention_mask", 
    "label_target_ids",
    "rationale_input_ids",
    "rationale_attention_mask",
    "rationale_target_ids"
])

## New Training Args
training_args = TrainingArguments(
    output_dir="./results",
    # Disable fp16 for MPS devices
    fp16=False,  # ← THIS IS CRUCIAL
    bf16=True,   # You can try enabling this if you have newer hardware
    use_mps_device=True,  # Explicitly enable MPS
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_dir="./logs",
    report_to="none",
    save_strategy="no",
    remove_unused_columns=False
)

new_trained_model_name = "distilled_t5_on_8194_samples"
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        alpha = 0.3  # λ hyperparameter from the paper
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Label Task ----------------------------------------------------------
        # Create decoder inputs by shifting labels
        label_decoder_input_ids = model._shift_right(inputs["label_target_ids"])
        
        # Process Label Task --------------------------------------------------
        label_outputs = model(
            input_ids=inputs["label_input_ids"],
            attention_mask=inputs["label_attention_mask"],
            decoder_input_ids=label_decoder_input_ids,
            return_dict=True
        )

        # Calculate loss for label prediction
        label_loss = ce_loss(
            label_outputs.logits.view(-1, model.config.vocab_size),
            inputs["label_target_ids"].view(-1)
        )

        # Process Rationale Task ----------------------------------------------
        rationale_outputs = model(
            input_ids=inputs["rationale_input_ids"],
            attention_mask=inputs["rationale_attention_mask"],
            labels=inputs["rationale_target_ids"]
        )
        rationale_loss = rationale_outputs.loss

        # Combine Losses ------------------------------------------------------
        total_loss = (1 - alpha) * label_loss + alpha * rationale_loss

        return (total_loss, (label_outputs, rationale_outputs)) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        alpha = 0.3

        label_inputs = {
            "input_ids": inputs["label_input_ids"],
            "attention_mask": inputs["label_attention_mask"],
            "labels": inputs["label_target_ids"]
        }

        rationale_inputs = {
            "input_ids": inputs["rationale_input_ids"],
            "attention_mask": inputs["rationale_attention_mask"],
            "labels": inputs["rationale_target_ids"]
        }

        # apparently super calls the parent class Trainer's prediction_step method 
        label_outputs = super().prediction_step(model, label_inputs, prediction_loss_only=False, ignore_keys=ignore_keys)
        rationale_outputs = super().prediction_step(model, rationale_inputs, prediction_loss_only=False, ignore_keys=ignore_keys)
        # this will reutrn (loss, logits, labels) which we can unpack
        label_loss, label_logits, label_labels = label_outputs
        rationale_loss, rationale_logits, rationale_labels = rationale_outputs
        # combine the losses now
        loss = (alpha * label_loss) + (1 - alpha) * rationale_loss

        if prediction_loss_only:
            return (loss, None, None)

        return (
            loss,
            [label_logits, rationale_logits],
            [label_labels, rationale_labels]
        )

# Initialize Trainer
trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=lambda data: {
        "label_input_ids": torch.stack([d["label_input_ids"] for d in data]),
        "label_attention_mask": torch.stack([d["label_attention_mask"] for d in data]),
        "label_target_ids": torch.stack([d["label_target_ids"] for d in data]),
        "rationale_input_ids": torch.stack([d["rationale_input_ids"] for d in data]),
        "rationale_attention_mask": torch.stack([d["rationale_attention_mask"] for d in data]),
        "rationale_target_ids": torch.stack([d["rationale_target_ids"] for d in data])
    }
)

trainer.train()
trainer.save_model(f"./{new_trained_model_name}")
tokenizer.save_pretrained(f"./{new_trained_model_name}")
