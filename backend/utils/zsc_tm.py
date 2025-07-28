import os
import json
import logging
import pandas as pd
from transformers import pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import APIRouter, Depends, HTTPException, Query
from utils.auth_utils import decode_jwt_token
from utils.common_functions_api import get_file_id_from_token
from models.db_models import ClassifyLabels
from sqlalchemy.orm import Session


# -------------------- Setup Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Core Classification Function --------------------
async def classify_documents():
    auth_scheme = HTTPBearer()
    REFRESH_URL = "http://localhost:8000/auth/refresh"
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
    refresh_token: str = Query(None)

    user_id, file_id = await get_file_id_from_token(
        credentials.credentials, refresh_token, REFRESH_URL
    )
    folder_path = f"uploads/{file_id}"
    documents = []

    # Load documents from files
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append(f.read().strip())
        elif filename.endswith(".csv"):
            df = pd.read_csv(filepath)
            documents.extend(df.iloc[:, 0].dropna().astype(str).tolist())
        elif filename.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                if isinstance(json_data, list):
                    documents.extend([str(item) for item in json_data])
                elif isinstance(json_data, dict):
                    documents.extend([str(value) for value in json_data.values()])
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append(f.read().strip())
        elif filename.endswith(".csv"):
            df = pd.read_csv(filepath)
            documents.extend(df.iloc[:, 0].dropna().astype(str).tolist())
        elif filename.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                if isinstance(json_data, list):
                    documents.extend([str(item) for item in json_data])
                elif isinstance(json_data, dict):
                    documents.extend([str(value) for value in json_data.values()])
        else:
            logger.warning(f"⚠️ Skipping unsupported file type: {filename}")

    if not documents:
        logger.error("❌ No valid documents found.")
        return

    logger.info("🚀 Starting Zero-Shot Classification...")

    candidate_labels = [
        "technology", "finance", "health", "education", "space", "climate",
        "ai", "economy", "science", "energy"
    ]

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    classified_results = []
    for doc in documents:
        result = classifier(doc, candidate_labels, multi_label=True)
        top_label = result["labels"][0]
        classified_results.append({"text": doc, "label": top_label})

    df_classified = pd.DataFrame(classified_results)
    logger.info(f"✅ Zero-Shot Classification Done:\n{df_classified}")

    logger.info("🔍 Starting Topic Modeling...")

    try:
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        topic_model = BERTopic(embedding_model=sentence_model, verbose=True)

        topics, probs = topic_model.fit_transform(df_classified["text"])
        df_classified["topic"] = topics

        logger.info("✅ BERTopic Modeling Done.")

        db: Session = Depends(get_db)

        user_labels = db.query(ClassifyLabels.label).filter(ClassifyLabels.user_id == user_id).all()
        label_set = set(label[0].lower() for label in user_labels)

        # Show results with final topic logic
        for i in range(len(documents)):
            text = df_classified.iloc[i]['text']
            top_label = df_classified.iloc[i]['label']
            topic_id = df_classified.iloc[i]['topic']

            if topic_id == -1:
                final_topic = top_label
                print(f"✅ Final Classification Topic: {final_topic} (from ZSC)")
            else:
                final_topic = f"BERTopic-{topic_id}"
                print(f"✅ Final Classification Topic: {final_topic} (from BERTopic)")

            match_result = 'y1' if final_topic.lower() in label_set else '-1'
            print(f"🔁 Match Result: {match_result}")

            print(f"[{i}] Text: {text}")
            print(f"    🔖 ZSC Label: {top_label}")
            print(f"    🧠 Topic ID: {topic_id}")
            print("")

        print("\n📊 Topic Info:\n", topic_model.get_topic_info())
        return match_result

    except Exception as e:
        logger.error(f"❌ Error in topic modeling: {e}")
        return []
