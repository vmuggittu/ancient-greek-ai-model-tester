from transformers import (
    #env,
    pipeline,
     AutoTokenizer,
     AutoModelForTokenClassification
)

# 1. Define Ancient Greek sample text (Homer, Iliad 1.1) , model, tokenizer, settings, pipeline
text = "μῆνιν ἄειδε, θεὰ, Πηληϊάδεω Ἀχιλῆος " #seeems to work better with this punctuation
model_name="models/greberta-finetuned-perseus"
#model_name="models/greberta-finetuned-perseus-onnx"

# 2. Define the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True,
    use_fast=True,
    add_prefix_space=False, # <--- VERY GOOD for RoBERTa-type models - BUT if =False RETURNS μῆνιν CORRECT AND ἄειδε WRONG, if =True THE OPPOSITE!
    model_max_length=128         # max limit acceepted from model
    #truncation=True,
    #clean_up_tokenization_spaces=True #NB: if clean_up_tokenization_spaces=True the tokenizer treats the submitted text as a single word :( but its behavior is irregular!
)

#  Force truncation settings directly on tokenizer - does not seem useful
#tokenizer.truncation = True
#tokenizer.padding = "max_length" # Optional: for fixed padding

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    local_files_only=True
)

ancient_greek_pos = pipeline(
    "token-classification",
    # local_files_only=True,
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy=None  # <--- IMPORTANT: keeps raw tokens
    #aggregation_strategy="simple"  # "simple", "first", "max", or "average"
)

# 3. Run the pipeline
results = ancient_greek_pos(text)

# 4. BYTE DECODING - necessary only in Python WITHOUT ONNX, and only if aggregation_strategy=None
for token_info in results:
    token_info['word'] = tokenizer.decode([tokenizer.convert_tokens_to_ids(token_info['word'])])

print(results)
