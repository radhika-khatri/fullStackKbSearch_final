from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from models.schemas import LabelRequest
from utils.auth_utils import decode_jwt_token

# --- Setup FastAPI ---
router = FastAPI()

# --- CORS for frontend access ---
router.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MongoDB setup ---
client = MongoClient("mongodb://localhost:27017/")
db = client["support_db"]
collection = db["ClassifyLabels"]

# --- Candidate Labels ---
CANDIDATE_LABELS = [
    "technology", "finance", "health", "education",
    "space", "climate", "economy", "science", "energy"
]

# --- GET endpoint to serve labels ---
@router.get("/labels")
def get_labels():
    return {"labels": CANDIDATE_LABELS}

# --- POST endpoint to save selected label with user_id ---
@router.post("/select-label")
def select_label(
    data: LabelRequest,
    request: Request
):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]
    try:
        payload = decode_jwt_token(token)
        user_id = payload.get("user_id")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

    if data.label not in CANDIDATE_LABELS:
        raise HTTPException(status_code=400, detail="Invalid label")

    collection.insert_one({
        "user_id": user_id,
        "label": data.label
    })

    return {"message": f"Label '{data.label}' stored successfully for user '{user_id}'"}
