import os
import numpy as np
import unicodedata
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
import torch

# ==============================================================================
# 1. DATA UPLOAD, XPOS VOCABULARY CREATION
# ==============================================================================

# Load dataset Universal Dependencies Perseus for Ancient Greek from Hugging Face
raw_datasets = load_dataset("universal-dependencies/universal_dependencies", "grc_perseus")

# XPOS tags in HF are raw strings, we have to create a static map of unique IDs
# Extract all the unique XPOS tags present in the three splits (train, dev, test)
unique_xpos = set()
for split in ["train", "dev", "test"]:
    for example in raw_datasets[split]:
        unique_xpos.update(example["xpos"])

# 1. Create ordered mapping dictionaries
xpos_list = sorted(list(unique_xpos))
label2id = {label: idx for idx, label in enumerate(xpos_list)}

# OPTIONAL but recommended: Add a fallback token for unknown/rare tags (UNK)
#if "UNK" not in label2id:
   # xpos_list.append("UNK")
    #label2id["UNK"] = len(label2id)

id2label = {idx: label for idx, label in enumerate(xpos_list)}
num_labels = len(xpos_list)

# Create String <-> ID conversion dictionaries
#label_list = sorted(list(unique_xpos))
#xpos_to_id = {tag: i for i, tag in enumerate(label_list)}
#id_to_xpos = {i: tag for i, tag in enumerate(label_list)}
xpos_to_id = label2id
id_to_xpos = id2label

print(f"Total number of unique XPOS tags detected in the Perseus dataset: {num_labels}")

# ==============================================================================
# 2. TOKENIZATION LOGIC AND XPOS ALIGNMENT (GreBerta bugfix)
# ==============================================================================
tokenizer = AutoTokenizer.from_pretrained("bowphs/GreBerta", use_fast=False)

# Tokenization function compatible with use_fast=False
def tokenize_and_align_labels_xpos(examples):
    # Initialize batch output lists
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    # Extract the special token IDs of the model (e.g. [CLS], [SEP])
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    # Iterate on every sentence of the batch
    for tokens_list, xpos_list_sentence in zip(examples["tokens"], examples["xpos"]):
        sentence_input_ids = [cls_token_id]
        sentence_labels = [-100]  #  [CLS] token takes -100

        # Tokenize each word individually
        for word, xpos_tag in zip(tokens_list, xpos_list_sentence):
            # Tokenize the single word (return a list of subtoken IDs without specials)
            word_subtoken_ids = tokenizer.encode(word, add_special_tokens=False)

            if not word_subtoken_ids:
                continue

            # Convert the XPOS string into its entire correspondent
            tag_id = label2id[xpos_tag]

            # Assign the real tag to the first subtoken of the word
            sentence_labels.append(tag_id)
            # Assign -100 to all remaining subtoken (subword split)
            sentence_labels.extend([-100] * (len(word_subtoken_ids) - 1))

            # Accumulate input_ids
            sentence_input_ids.extend(word_subtoken_ids)

        # Close the sentence with the [SEP] token
        sentence_input_ids.append(sep_token_id)
        sentence_labels.append(-100)

        # Manual truncation management (if the sentence exceeds the maximum length, e.g. 512)
        max_len = tokenizer.model_max_length
        if len(sentence_input_ids) > max_len:
            sentence_input_ids = sentence_input_ids[:max_len]
            sentence_labels = sentence_labels[:max_len]

        # Create the attention mask (all 1 for the real tokens entered)
        attention_mask = [1] * len(sentence_input_ids)

        # Save into respective batch
        batch_input_ids.append(sentence_input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(sentence_labels)

    # Return the dictionary expected from Hugging Face datasets.map()
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }

# Apply structured XPOS mapping
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels_xpos,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

# ==============================================================================
# 3. EVALUATION/RATING METRICS WITH XPOS STRING RESTORATION
# ==============================================================================

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    # convert the predicted and real IDs back into the original XPOS strings
    # ignoring tokens masked with -100
    true_predictions = [
        [id_to_xpos[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_xpos[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# ==============================================================================
# 4. MODEL CONFIGURATION WITH MAP DICTIONARIES AND TRAINER
# ==============================================================================

# initialize the model by explicitly passing the high number of XPOS labels
# and id2label/label2id maps to allow the model to save the correct metadata
model = AutoModelForTokenClassification.from_pretrained(
    "bowphs/GreBerta",
    num_labels=num_labels,
    id2label=id_to_xpos,
    label2id=xpos_to_id
)

training_args = TrainingArguments(
    output_dir="./greberta-xpos-perseus",
    learning_rate=5e-5,               #LR slightly higher to manage the wider label space
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    tokenizer=tokenizer, #deprecated. use:  processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start morphological fine-tuning with high granularity
trainer.train()


#8. Optimized model saving in PyTorch
print("Optimized model saving...")
model.save_pretrained("./greberta-finetuned")
tokenizer.save_pretrained("./greberta-finetuned")

#9. Automatic export into ONNX format required by transformers.js
print("Export into ONNX format for JavaScript...")
os.system("./optimum-cli export onnx --model ./greberta-finetuned ./greberta-onnx/")
print("Process completed! Files ready in './greberta-onnx/'")
