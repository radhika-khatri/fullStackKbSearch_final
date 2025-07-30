# ==========================================
# 💡 Support Assistant Project .gitignore
# For FastAPI + PostgreSQL + React + Docker
# Author: Shivam Srivastav
# ==========================================
# File: backend/api/files.py    
# ==========================================
from requests import Session
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pathlib import Path
from utils.embed_store import chunk_text, save_embeddings
from utils.auth_utils import decode_jwt_token
from jose.exceptions import ExpiredSignatureError
import httpx
import os
from sqlalchemy import select

from database.database import get_db
from models.db_models import User
from sqlalchemy.ext.asyncio import AsyncSession

from training.fine_tune import fine_tune
from utils.file_parser import parse_file, clean_json, extract_full_text
from utils.email_notify import send_upload_notification
from utils.zsc_tm import classify_documents
from logging import getLogger

logger = getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

auth_scheme = HTTPBearer()
REFR_URL = "http://localhost:8000/auth/refresh"

ALLOWED_EXTENSIONS = {".csv", ".json", ".xlsx", ".txt", ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp"}
MAX_FILE_SIZE_MB = 10


@router.post("/upload", summary="Upload training/WhatsApp file")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
    refresh_token: str = Query(None),
    db: Session = Depends(get_db)
):
    # ─── JWT Decode ───
    try:
        credentials = credentials.credentials  # Unpacking the tuple

        try:
            payload = decode_jwt_token(credentials)
        except ExpiredSignatureError:
            if not refresh_token:
                raise HTTPException(status_code=401, detail="Token expired. Provide refresh_token.")
            async with httpx.AsyncClient() as client:
                res = await client.post(REFR_URL, json={"refresh_token": refresh_token})
                if res.status_code != 200:
                    raise HTTPException(status_code=401, detail="Refresh token invalid")
                new_token = res.json()["token"]
                payload = decode_jwt_token(new_token)

        user_email = payload.get("sub", "anonymous")  # using email (string)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    # ─── File Type & Size Check ───
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(contents)

    # ─── Parse Preview ───
    try:
        preview = parse_file(file_path, limit=5)
        preview = clean_json(preview)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")

    # ─── Embedding and Save ───
    match_result = await classify_documents(file_path, db)  # Pass file_path to the function
    print(f"Match Result: {match_result}")
    if match_result == '-1':
        raise HTTPException(status_code=400, detail="Incorrect information provided. Cannot train model.")
    
    else:
        try:
            print("You have provided correct info about the file. Proceeding with embedding generation...")
            full_text = parse_file(file_path)
            if isinstance(full_text, list):
                full_text = " ".join(
                    str(item.get("prompt", "")) + " " + str(item.get("response", ""))
                    for item in full_text if isinstance(item, dict)
                )
            elif isinstance(full_text, dict):
                full_text = " ".join(str(v) for v in full_text.values())

            elif not isinstance(full_text, str):
                full_text = str(full_text)

            # ─── Chunk and Save Embeddings ───
            chunks = chunk_text(full_text)
            if chunks:
                save_embeddings(file_id=file.filename, chunks=chunks)
            else:
                raise HTTPException(status_code=500, detail="❌ No valid chunks generated.")
        except Exception as e:
            print(f"[Embedding Error]: {e}")
            raise HTTPException(status_code=500, detail=f"❌ Embedding generation failed: {str(e)}")

    # ─── Update user's file_id in DB ───
    try:
        result = await db.execute(select(User).where(User.email == user_email))
        user = result.scalar_one_or_none()
        if user:
            user.file_id = file.filename  # Store filename as file_id
            await db.commit()
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update user with file_id: {str(e)}")

    # ─── Optional: Background fine-tune ───
    # background_tasks.add_task(fine_tune)

    # ─── Email Notification ───
    send_upload_notification("vastavshivam@gmail.com", "uploaded successfully.", body="File uploaded ...")

    # ─── Return Response ───
    return JSONResponse(content={
        "message": f"File {file.filename} uploaded successfully.",
        "filename": file.filename,
        "preview": preview,
        "uploaded_by": user_email,
        "label": match_result
    })
