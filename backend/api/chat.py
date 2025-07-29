import os
import pickle
import logging
import faiss
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from jose.exceptions import ExpiredSignatureError
import httpx

from utils.auth_utils import decode_jwt_token
from utils.nlp import analyze_sentiment, compute_embedding
from utils.common_functions_api import get_file_id_from_token  # <- ✅ new
from models.schemas import ChatMessage
from db.crud import save_chat_message, get_conversation_context
from services.rag import generate_response_with_rag
import logging

# Optional: sentiment and embedding generators
from utils.nlp import analyze_sentiment, compute_embedding

logger = logging.getLogger(__name__)
router = APIRouter()
auth_scheme = HTTPBearer()
REFRESH_URL = "http://localhost:8000/auth/refresh"
VECTOR_DIR = "vector_stores"
MODEL_PATH = os.path.join(VECTOR_DIR, "falcon-rw-1b")

# ───── LOGGING ─────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───── Load Vector Model ─────
vector_model = SentenceTransformer("all-MiniLM-L6-v2")

# ───── Load Language Model ─────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ───── API Models ─────
class ChatRequest(BaseModel):
    query: str


# ───── CHAT ENDPOINT ─────
@router.post("/chat")
async def chat(
    msg: ChatMessage,
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
    refresh_token: str = Query(None)
):
    try:
        user_id, file_id = await get_file_id_from_token(
            credentials.credentials, refresh_token, REFRESH_URL
        )

        index_path = os.path.join(VECTOR_DIR, f"{file_id}.index")
        chunks_path = os.path.join(VECTOR_DIR, f"{file_id}_chunks.pkl")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise HTTPException(status_code=404, detail=f"Vector files not found for file_id: {file_id}")

        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            documents = pickle.load(f)

        query = msg.message
        logger.info(f"🟢 Query: {query}")

        query_vector = vector_model.encode([query]).astype("float32").reshape(1, -1)
        distances, indices = index.search(query_vector, k=1)
        top_context = documents[indices[0][0]]
        logger.info(f"📄 Context: {top_context[:200]}...")

        prompt = f"<s>Context: {top_context}\nCustomer: {query}\nAgent:"
        logger.info(f"💬 Prompt: {prompt}")
        response = chatbot(prompt, max_length=200, do_sample=True)[0]["generated_text"]
        logger.info(f"🤖 Response: {response}")
        reply = response.split("Agent:")[-1].strip()

        sentiment = analyze_sentiment(query)
        embedding = compute_embedding(query)

        save_chat_message(
            user_id=user_id,
            session_id=msg.session_id,
            message=query,
            sender="user",
            sentiment=sentiment,
            embedding=embedding,
            bot_reply=reply
        )

        return {"reply": reply}

    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Chat failed")


# ───── WebSocket API ─────
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    token = websocket.headers.get("Authorization")
    refresh_token = websocket.headers.get("X-Refresh-Token")

    if token is None or not token.startswith("Bearer "):
        await websocket.close(code=1008, reason="Missing or invalid Authorization header")
        return

    try:
        user_id, file_id = await get_file_id_from_token(
            token[7:], refresh_token, REFRESH_URL
        )

        index_path = os.path.join(VECTOR_DIR, f"{file_id}.index")
        chunks_path = os.path.join(VECTOR_DIR, f"{file_id}_chunks.pkl")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            await websocket.send_text("Vector files not found.")
            await websocket.close(code=1011)
            return

        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            documents = pickle.load(f)

        while True:
            msg_text = await websocket.receive_text()
            sentiment = analyze_sentiment(msg_text)
            embedding = compute_embedding(msg_text)

            query_vector = vector_model.encode([msg_text]).astype("float32").reshape(1, -1)
            distances, indices = index.search(query_vector, k=1)
            top_context = documents[indices[0][0]]

            prompt = f"<s>Context: {top_context}\nCustomer: {msg_text}\nAgent:"
            response = chatbot(prompt, max_length=200, do_sample=True)[0]["generated_text"]
            reply = response.split("Agent:")[-1].strip()

            save_chat_message(
                user_id=user_id,
                session_id="ws",
                message=msg_text,
                sender="user",
                sentiment=sentiment,
                embedding=embedding,
                bot_reply=reply
            )

            await websocket.send_text(reply)

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {user_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        await websocket.send_text("Something went wrong.")
        await websocket.close(code=1011)


# ───── Conversation History ─────
@router.get("/history/{session_id}")
def get_history(
    session_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
    refresh_token: str = Query(None)
):
    try:
        payload = decode_jwt_token(credentials.credentials)
    except ExpiredSignatureError:
        if not refresh_token:
            raise HTTPException(status_code=401, detail="Token expired. Provide refresh_token.")
        with httpx.Client() as client:
            res = client.post(REFRESH_URL, json={"refresh_token": refresh_token})
            if res.status_code != 200:
                raise HTTPException(status_code=401, detail="Refresh token invalid")
            new_token = res.json()["token"]
            payload = decode_jwt_token(new_token)

    return get_conversation_context(session_id)
