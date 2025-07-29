# ==========================================
# 💡 Support Assistant Project .gitignore
# For FastAPI + PostgreSQL + React + Docker
# team: AppgBird
# Author: Shivam Srivastav
# ==========================================
# File: backend/utils/whatsapp_api.py
# This module handles WhatsApp Cloud API integration for sending and receiving messages. 
# ==========================================
import os
import numpy as np
import cv2
import pytesseract
from fastapi import Request, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models.db_models import ClientConfig
from database.database import get_db
from typing import List, Optional
import httpx
import dotenv
import ollama
import aiohttp
import logging  
import requests
import json

from fastapi import Request, HTTPException
from models.db_models import ChatHistory
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import dotenv
from utils.service import is_valid_gst
from utils.service import extract_raw_text_from_image, preprocess_image, extract_gst_number, extract_relevant_info_with_mistral, parse_mistral_response
                           


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

CURRENT_API_VERSION = os.getenv("CURRENT_API_VERSION", "v23.0")  # Default to v23.0 if not set

# Define the base URL for WhatsApp API


BASE_URL = f"https://graph.facebook.com/{CURRENT_API_VERSION}"
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID") 
WHATSAPP_API_URL = f"https://graph.facebook.com/{CURRENT_API_VERSION}/{WHATSAPP_PHONE_ID}"



# ✅ Verify Webhook Token (GET)
def verify_webhook(mode: str, token: str, challenge: str):
    VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
    
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return int(challenge)
    raise HTTPException(status_code=403, detail="Webhook verification failed")

# 📩 Handle Incoming Webhook Messages (POST)
# def handle_incoming_message(req: Request):
#     payload = req.json()
#     entry = payload.get("entry", [])[0]
#     changes = entry.get("changes", [])[0]
#     value = changes.get("value", {})
#     messages = value.get("messages", [])

#     if messages:
#         message = messages[0]
#         from_id = message["from"]
#         text = message.get("text", {}).get("body")

#         # Log or store the message
#         print(f"Incoming WhatsApp message from {from_id}: {text}")
#         # Optional: save to DB and respond using AI or fallback

#     return {"status": "received"}



async def download_image(url: str, token: str) -> bytes:
    headers = {"Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                return await resp.read()
            raise Exception(f"Failed to download image: {resp.status}")

def extract_text_from_image(image_bytes: bytes) -> str:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    text = pytesseract.image_to_string(img_np)
    return text.strip()

async def handle_incoming_message(req: Request, client_config: dict, db=None):
    count = 0
    
    payload = await req.json()
    entry = payload.get("entry", [])[0]
    changes = entry.get("changes", [])[0]
    value = changes.get("value", {})
    messages = value.get("messages", [])
    metadata = value.get("metadata", {})
    phone_id = metadata.get("phone_number_id")

    if messages:
        message = messages[0]
        from_id = message["from"]
        client_id = None

        # Match client config by phone ID
        for cid, config in client_config.items():
            print(f"Checking config for {cid}: {config['phone_id']}")
            if config["phone_id"] == phone_id:
                client_id = cid
                print(f"Matched client ID: {client_id}")
                break

        # Handle text message
        text = message.get("text", {}).get("body")
        msg_type = message.get("type")
        print(f"Received message from {from_id}: {text} (Type: {msg_type})")

        if msg_type == "text" and text:
            print(f"Text from {from_id}: {text}")

            if db and client_id:
                try:
                    db.add(ChatHistory(
                        client_id=client_id,
                        sender="user",
                        user_number=from_id,
                        message=text,
                        message_type=msg_type,
                        direction="incoming"
                    ))
                    await db.commit()
                except Exception as db_err:
                    await db.rollback()
                    print(f"❌ Error inserting incoming chat: {db_err}")
                    
            
            reply = f"🤖 Bot: hi sir we will connect you soon '{from_id}'"

        # Handle image message
        elif msg_type == "image":
            image_id = message["image"]["id"]
            config = client_config[client_id]
            media_url_resp = await get_whatsapp_media_url(image_id, config["token"])
            image_bytes = await download_image(media_url_resp, config["token"])
            # image_bytes = await download_image(media_url_resp, config["token"])
            raw_text = extract_raw_text_from_image(image_bytes)
            
            gst_number = extract_gst_number(raw_text)
            valid_gst = is_valid_gst(gst_number)

            if not gst_number or not valid_gst:
                return {
                    "message": "GST number not found or invalid. Please enter your GST number manually."
                }

            # 🤖 Call Mistral for additional info
            mistral_response = extract_relevant_info_with_mistral(raw_text)
            parsed_info = parse_mistral_response(mistral_response)

            if db and client_id:
                try:
                    db.add(ChatHistory(
                        client_id=client_id,
                        sender="user",
                        user_number=from_id,
                        message=text,
                        message_type=msg_type,
                        direction="incoming"
                    ))
                    await db.commit()
                except Exception as db_err:
                    await db.rollback()
                    print(f"❌ Error inserting incoming chat: {db_err}")

            # reply = f"🖼️ I read this from your image: '{extracted_text}'"
            reply = f"🖼️ I read this from your image: 'extracted_text'"

        else:
            reply = "❓ Sorry, I only understand text or image messages for now."

        # Send reply
        if client_id and reply:
            config = client_config[client_id]
            await send_whatsapp_message_dynamic(
                token=config["token"],
                phone_id=config["phone_id"],
                to=from_id,
                message=reply
            )

            if db:
                try:
                    db.add(ChatHistory(
                        client_id=client_id,
                        sender="bot",
                        user_number=from_id,
                        message=reply,
                        message_type="text",  # Assuming reply is always text for now
                        direction="outgoing"
                    ))
                    await db.commit()
                except Exception as db_err:
                    await db.rollback()
                    print(f"❌ Error inserting bot reply: {db_err}")

    return {"status": "received"}


# 📤 Send WhatsApp Message via Cloud API
def send_whatsapp_message(to: str, message: str):
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    response = requests.post(WHATSAPP_API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        print("WhatsApp API Error:", response.text)
    return response.json()

def send_whatsapp_message_dynamic(token: str, phone_id: str, to: str, message: str):
    # url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
    url = f"{BASE_URL}/{phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        print("WhatsApp API Error:", response.text)
    return response.json()


# async def send_whatsapp_message_dynamic(token, phone_id, to, message):
#     url = f"{BASE_URL}/{phone_id}/messages"
#     headers = {
#         "Authorization": f"Bearer {token}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "messaging_product": "whatsapp",
#         "to": to,
#         "type": "text",
#         "text": {"body": message}
#     }
#     async with httpx.AsyncClient() as client:
#         response = await client.post(url, json=payload, headers=headers)
#         return response.json()

async def get_whatsapp_media_url(media_id: str, token: str) -> str:
    url = f"https://graph.facebook.com/v19.0/{media_id}"
    headers = {"Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("url")
            else:
                raise Exception(f"Failed to fetch media URL: {resp.status}")

async def mark_message_as_read(token, message_id):
    url = f"{BASE_URL}/messages"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"messaging_product": "whatsapp", "status": "read", "message_id": message_id}
    async with httpx.AsyncClient() as client:
        return (await client.post(url, json=payload, headers=headers)).json()
    




async def send_template_message(token, phone_id, to, template_name, language_code, components=None):
    url = f"{BASE_URL}/{phone_id}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language_code}
        }
    }
    async with httpx.AsyncClient() as client:
        return (await client.post(url, json=payload, headers=headers)).json()
    
async def get_message_status(token, message_id):
    url = f"{BASE_URL}/{message_id}"
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        return (await client.get(url, headers=headers)).json()

async def get_business_profile(token, phone_id):
    url = f"{BASE_URL}/{phone_id}/whatsapp_business_profile"
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        return (await client.get(url, headers=headers)).json()

async def subscribe_webhook(app_id, token, webhook_url, verify_token):
    url = f"https://graph.facebook.com/v19.0/{app_id}/subscriptions"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "object": "whatsapp_business_account",
        "callback_url": webhook_url,
        "verify_token": verify_token,
        "fields": "messages"
    }
    async with httpx.AsyncClient() as client:
        return (await client.post(url, data=payload, headers=headers)).json()
    

async def update_profile(token, phone_id, profile_data):
    url = f"{BASE_URL}/{phone_id}/whatsapp_business_profile"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        return (await client.put(url, json=profile_data, headers=headers)).json()



