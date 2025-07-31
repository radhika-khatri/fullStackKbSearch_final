from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pinecone import Pinecone, ServerlessSpec
from gensim.models import Word2Vec
import numpy as np
import uuid
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")

# ------------------- Load Models -------------------
print("[🧠] Loading models...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

summarizer_model_name = "facebook/bart-large-cnn"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name)

# ------------------- MongoDB -------------------
print("[🔌] Connecting to MongoDB...")

client = MongoClient("mongodb+srv://RootAdmin:root@atlascluster.0ktshci.mongodb.net/?retryWrites=true&w=majority&appName=AtlasCluster")
collection = client["Portfolio"]["Portfolio-Website"]

# ------------------- Pinecone -------------------
print("[🌐] Connecting to Pinecone...")

pc = Pinecone(api_key="***")  # Replace with your actual API key
index_name = "pdf-rag-index"
w2v_index_name = "word2vec-review-index"

for name, dim in [(index_name, 384), (w2v_index_name, 100)]:
    if name not in pc.list_indexes().names():
        print(f"[📦] Creating index '{name}'...")
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

index = pc.Index(index_name)
w2v_index = pc.Index(w2v_index_name)

# ------------------- Train Word2Vec -------------------
print("[📖] Gathering sentences for Word2Vec training...")

all_sentences = []
for session in collection.find():
    chat = session.get("chat", [])
    for msg in chat:
        tokens = word_tokenize(msg["message"].lower())
        all_sentences.append(tokens)

print("[🧠] Training Word2Vec model...")
w2v_model = Word2Vec(sentences=all_sentences, vector_size=100, window=5, min_count=1, workers=4)

# ------------------- Process Chat Sessions -------------------
print("[🚀] Starting chat session processing...")

for session in collection.find():
    session_id = str(session["_id"])
    chat = session.get("chat", [])

    if not chat:
        print(f"[⚠️] Skipping empty session: {session_id}")
        continue

    print(f"\n🔍 Processing session: {session_id}")
    chat_text = " ".join([msg["message"] for msg in chat])
    print(f"[💬] Chat text length: {len(chat_text)} characters")

    try:
        # 1. Sentiment Analysis
        chunks = [chat_text[i:i+512] for i in range(0, len(chat_text), 512)]
        sentiments = [sentiment_pipeline(chunk)[0] for chunk in chunks]
        avg_score = float(np.mean([s["score"] for s in sentiments]))
        overall_label = max(set(s["label"] for s in sentiments), key=[s["label"] for s in sentiments].count)
        print(f"[❤️] Sentiment: {overall_label} | Avg Score: {avg_score:.2f}")

        # 2. Summary Generation
        print("[✍️] Generating summary...")
        inputs = summarizer_tokenizer([chat_text], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = summarizer_model.generate(
            inputs["input_ids"], max_length=150, min_length=30,
            length_penalty=2.0, num_beams=4, early_stopping=True
        )
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(f"[📝] Summary: {summary}")

        # 3. Update MongoDB
        collection.update_one(
            {"_id": session["_id"]},
            {"$set": {
                "overall_sentiment": overall_label,
                "sentiment_score": avg_score,
                "chat_summary": summary
            }}
        )
        print("[✅] MongoDB updated.")

        # 4. Pinecone (SentenceTransformer)
        pinecone_id = str(uuid.uuid4())
        sentence_vec = embedder.encode(summary).tolist()
        index.upsert([(pinecone_id, sentence_vec, {
            "session_id": session_id,
            "summary": summary,
            "sentiment": overall_label,
            "sentiment_score": avg_score,
            "source": "sentence-transformer"
        })])
        print(f"[📅] SentenceTransformer vector stored with ID: {pinecone_id}")

        # 5. Pinecone (Word2Vec)
        tokens = word_tokenize(chat_text.lower())
        valid_vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if valid_vectors:
            w2v_vector = np.mean(valid_vectors, axis=0).tolist()
        else:
            w2v_vector = [0.0] * w2v_model.vector_size

        w2v_id = pinecone_id + "-w2v"
        w2v_index.upsert([(w2v_id, w2v_vector, {
            "session_id": session_id,
            "source": "word2vec"
        })])
        print(f"[📅] Word2Vec vector stored with ID: {w2v_id}")

    except Exception as e:
        print(f"[❌] Error in session {session_id}: {e}")

print("\n[🏋️] All chat sessions processed and indexed.")
