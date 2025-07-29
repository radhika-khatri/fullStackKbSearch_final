from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import MODEL_NAME, OUTPUT_DIR
import torch
import os

# Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR or MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR or MODEL_NAME)

# model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
# tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")

# model.save_pretrained("uploaded_kbs/falcon-rw-1b")
# tokenizer.save_pretrained("uploaded_kbs/falcon-rw-1b")

# Use local directory only
model_path = OUTPUT_DIR if os.path.exists(OUTPUT_DIR) else MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = model.to(device)

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def get_mistral_response(prompt: str, max_new_tokens: int = 100) -> str:
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()


def classify_sentiment(text: str) -> str:
    """
    Returns 'positive' or 'negative'
    """
    result = classifier(text)[0]
    label = result["label"].lower()  # 'positive' or 'negative'
    return label


# def get_mistral_response(prompt: str, max_new_tokens: int = 100) -> str:
#     inputs = tokenizer.encode(prompt, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs,
#             max_new_tokens=max_new_tokens,
#             pad_token_id=tokenizer.eos_token_id,
#             do_sample=True,
#             top_k=50,
#             top_p=0.95
#         )
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return decoded[len(prompt):].strip()