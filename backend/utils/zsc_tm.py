import os
import json
import logging
import pandas as pd
from sqlalchemy import select
from fastapi import APIRouter, Depends
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

from database.database import get_db
from models.db_models import ClassifyLabels
from sqlalchemy.orm import Session

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Initialize router
router = APIRouter()

@router.get("/classify-documents")
async def classify_documents(file_path: str, db: Session = Depends(get_db)):
    filename = file_path
    documents = []

    # --- Load documents ---
    if filename.endswith(".txt"):
        with open(filename, "r", encoding="utf-8") as f:
            documents.append(f.read().strip())
    elif filename.endswith(".csv"):
        df = pd.read_csv(filename)
        documents.extend(df.iloc[:, 0].dropna().astype(str).tolist())
    elif filename.endswith(".json"):
        with open(filename, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            if isinstance(json_data, list):
                documents.extend([str(item) for item in json_data])
            elif isinstance(json_data, dict):
                documents.extend([str(value) for value in json_data.values()])
    else:
        logger.warning(f"⚠️ Skipping unsupported file type: {filename}")

    if not documents:
        logger.error("❌ No valid documents found.")
        return "-1"

    logger.info("🚀 Starting Zero-Shot Classification...")

    candidate_labels = [
        "technology", "finance", "health", "education", "space", "climate",
        "economy", "science", "energy"
    ]

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device='cpu')

    # Classify each document
    classified_results = []
    for doc in documents:
        result = classifier(doc, candidate_labels, multi_label=True)
        top_label = result["labels"][0]
        classified_results.append({"text": doc, "label": top_label})

    logger.info("✅ Zero-Shot Classification Done.")

    # Load user labels from DB
    user_id = "admin123@gmail.com"
    result = await db.execute(select(ClassifyLabels.label).where(ClassifyLabels.user_id == user_id))
    user_labels = result.scalars().all()
    label_set = set(label.lower() for label in user_labels)

    if not label_set:
        logger.warning("❌ No user-defined labels found.")
        return "-1"

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    similarities = []

    for item in classified_results:
        text = item["text"]
        predicted_label = item["label"]

        logger.info(f"📄 Text: {text}")
        logger.info(f"🔖 Predicted Label: {predicted_label}")

        # Compute embedding for predicted label
        pred_embedding = sentence_model.encode(predicted_label.lower())

        # Find max similarity with any user label
        max_sim = -1
        for user_label in label_set:
            user_embedding = sentence_model.encode(user_label)
            sim = cosine_similarity(pred_embedding, user_embedding)
            max_sim = max(max_sim, sim)

        logger.info(f"🔍 Max similarity with user labels: {max_sim:.4f}")
        similarities.append(max_sim)

    # Compute average similarity
    avg_similarity = sum(similarities) / len(similarities)
    logger.info(f"📊 Average similarity across all documents: {avg_similarity:.4f}")

    final_result = 'y1' if avg_similarity >= 0.6 else '-1'
    logger.info(f"✅ Final Match Result: {final_result}")
    return final_result
