from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss, pickle, json, os
import logging
from transformers import TextStreamer
import torch

# --- CONFIGURATION ---
MODEL_PATH = os.path.abspath("checkpoints/fine-tuned-output")  # Path to your fine-tuned model
VECTOR_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "kb.index"
DOCS_PATH = "kb_docs.pkl"
TRAINING_DATA_PATH = os.path.abspath("training/data/tune.jsonl")

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


router = APIRouter()

# --- Load Fine-Tuned Mistral Model ---
logger.info("📦 Loading fine-tuned Mistral model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, device=torch.device("cpu"))

# --- Load Embedding Model ---
embedder = SentenceTransformer(VECTOR_MODEL_NAME)

# --- Vector DB Initialization ---
print(f"🔄 Loading vector DB from: {INDEX_PATH}======={TRAINING_DATA_PATH}")
def build_vector_index_from_jsonl(jsonl_path=TRAINING_DATA_PATH):
    logger.info("📖 Building vector index from JSONL...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        documents = []
        for line in f:
            item = json.loads(line)
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            documents.append(f"{prompt}\n{response}")

    vectors = embedder.encode(documents)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    logger.info("✅ Vector index and document store saved.")

# --- Load or Build Vector DB ---
if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
    build_vector_index_from_jsonl()

logger.info("📂 Loading vector DB...")
index = faiss.read_index(INDEX_PATH)
with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)

# --- API MODEL ---
class ChatRequest(BaseModel):
    query: str

def generate_clean_response(prompt, model, tokenizer):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Move to GPU if available
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        model = model.cuda()

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        do_sample=False,                         # Deterministic output
        eos_token_id=tokenizer.eos_token_id,     # Stop at </s>
        pad_token_id=tokenizer.eos_token_id,     # Avoid pad token warning
        repetition_penalty=1.5,                  # Penalize repeating phrases
        no_repeat_ngram_size=3                   # Prevent 3-gram repeats
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Only keep what comes after 'Agent:'
    if "Agent:" in output_text:
        return output_text.split("Agent:")[-1].strip()
    return output_text.strip()
# --- /chat ENDPOINT ---
@router.post("/vchat")
def chat(request: ChatRequest):
    query = request.query
    logger.info(f"🟢 Received query: {query}")

    # Vector search
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec, k=1)
    context = documents[indices[0][0]]
    logger.info(f"📄 Retrieved context: {context}")

    # Build prompt
    prompt = f"<s>Customer: {query}\nAgent:"

    # Clean generation
    response = generate_clean_response(prompt, model, tokenizer)
    logger.info(f"💬 Generated response: {response}")

    return {"response": response}
