from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoModelForSeq2SeqLM
from pymongo.errors import PyMongoError
import numpy as np
import json

print("[🔌] Connecting to MongoDB...")

# MongoDB connection
client = MongoClient("mongodb+srv://RootAdmin:Root@atlascluster.0ktshci.mongodb.net/?retryWrites=true&w=majority&appName=AtlasCluster")
db = client["Portfolio"]
collection = db["Portfolio-Website"]

print("[✅] MongoDB connected.")
print("[📦] Loading sentiment model...")

# Sentiment model
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

print("[📚] Loading summarization model...")
# Summarization model
summarizer_model_name = "facebook/bart-large-cnn"
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name)
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)

print("[✅] All models loaded.")

def format_chat_session(chat):
    return " [SEP] ".join([f"{msg['sender'].upper()}: {msg['message']}" for msg in chat])

def chunk_text_by_tokens(text, max_tokens=512):
    tokens = sentiment_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = sentiment_tokenizer.decode(tokens[i:i+max_tokens])
        chunks.append(chunk)
    return chunks

def summarize_chat(text):
    inputs = summarizer_tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = summarizer_model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("[🚀] Starting sentiment analysis and summarization...")

total, processed, skipped = 0, 0, 0

for session in collection.find():
    total += 1
    session_id = session["_id"]
    chat = session.get("chat", [])

    print(f"\n---\n🧾 Session ID: {session_id}")
    
    if not chat:
        print("⚠️  Skipped: Empty chat session.")
        skipped += 1
        continue

    try:
        chat_text = format_chat_session(chat)
        chunks = chunk_text_by_tokens(chat_text)

        all_scores = []
        all_labels = []

        for i, chunk in enumerate(chunks):
            result = sentiment_pipeline(chunk)[0]
            print(f"[🔍] Chunk {i+1}: {result['label']} ({result['score']:.2f})")
            all_scores.append(result["score"])
            all_labels.append(result["label"])

        final_label = max(set(all_labels), key=all_labels.count)
        avg_score = float(np.mean(all_scores))
        summary = summarize_chat(chat_text)

        # Update MongoDB
        collection.update_one(
            {"_id": session_id},
            {"$set": {
                "overall_sentiment": final_label,
                "sentiment_score": avg_score,
                "chat_summary": summary
            }}
        )

        # Pretty print output
        summary_dict = {
            "_id": str(session_id),
            "chat": "[...]",
            "overall_sentiment": final_label,
            "sentiment_score": round(avg_score, 2),
            "chat_summary": summary
        }

        print("\n📝 Final Session Summary:")
        print(json.dumps(summary_dict, indent=2))
        processed += 1

    except PyMongoError as db_err:
        print(f"❌ MongoDB Error: {db_err}")
        skipped += 1
    except Exception as e:
        print(f"❌ Error processing session {session_id}: {e}")
        skipped += 1

print("\n---\n🎯 Analysis Summary:")
print(f"Total sessions found: {total}")
print(f"Processed successfully: {processed}")
print(f"Skipped (empty or error): {skipped}")
