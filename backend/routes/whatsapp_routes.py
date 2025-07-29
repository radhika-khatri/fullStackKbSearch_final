from fastapi import APIRouter, Depends, HTTPException, Request , status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from utils import whatsapp_api
from models.db_models import ClientConfig
from database.database import get_db
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from typing import Optional, Literal



router = APIRouter()


class WhatsAppConfig(BaseModel):
    client_id: str
    phone_id: str
    token: str
    verify_token: str

class SendMessagePayload(BaseModel):
    client_id: str
    to: str
    message: str

class TemplateComponent(BaseModel):
    type: str
    parameters: List[dict]

class TemplateMessagePayload(BaseModel):
    client_id: str
    to: str
    template_name: str
    language_code: str
    components: Optional[List[TemplateComponent]] = None

class MarkReadPayload(BaseModel):
    client_id: str
    message_id: str

class UpdateProfilePayload(BaseModel):
    client_id: str = Field(..., description="Unique client identifier")
    business_name: Optional[str] = Field(None, description="Business display name")
    about: Optional[str] = Field(None, description="Business about info")
    email: Optional[str] = Field(None, description="Business contact email")
    website: Optional[str] = Field(None, description="Business website")
    address: Optional[str] = Field(None, description="Business address")
    vertical: Optional[str] = Field(None, description="Business vertical e.g., 'RETAIL', 'EDUCATION'")

    class Config:
        schema_extra = {
            "example": {
                "client_id": "client_123",
                "business_name": "My Company",
                "about": "We sell gadgets and accessories.",
                "email": "support@mycompany.com",
                "website": "https://mycompany.com",
                "address": "123, Business Street, NY",
                "vertical": "RETAIL"
            }
        }

class MediaMessagePayload(BaseModel):
    messaging_product: Literal["whatsapp"]
    to: str
    type: Literal["image", "audio", "video", "document", "sticker"]
    image: Optional[dict] = None
    audio: Optional[dict] = None
    video: Optional[dict] = None
    document: Optional[dict] = None
    sticker: Optional[dict] = None

    class Config:
        schema_extra = {
            "example": {
                "messaging_product": "whatsapp",
                "to": "919876543210",
                "type": "image",
                "image": {
                    "link": "https://example.com/path/to/image.jpg",
                    "caption": "Check this out!"
                }
            }
        }


# helper.py or within your router function
def get_token_and_phone_id_by_phone(phone_number: str, db: Session):
    config = db.query(ClientConfig).filter(ClientConfig.phone_number == phone_number).first()
    if not config:
        raise ValueError(f"No config found for phone number: {phone_number}")
    return config.token, config.phone_id

# helpers.py

async def get_token_and_phone_id(client_id: str, db):
    result = await db.execute(
        select(ClientConfig).where(ClientConfig.client_id == client_id)
    )
    config = result.scalars().first()
    if not config:
        raise ValueError(f"No config found for client_id: {client_id}")
    return config.token, config.phone_id



# @router.post("/configure-whatsapp")
# def configure_whatsapp(config: WhatsAppConfig, db: Session = Depends(get_db)):
    
#     existing = db.query(ClientConfig).filter(ClientConfig.client_id == config.client_id).first()
#     if existing:
#         existing.phone_id = config.phone_id
#         existing.token = config.token
#         existing.verify_token = config.verify_token
#     else:
#         new_config = ClientConfig(
#             client_id=config.client_id,
#             phone_id=config.phone_id,
#             token=config.token,
#             verify_token=config.verify_token
#         )
#         db.add(new_config)
#     db.commit()
#     return {"message": "WhatsApp config saved/updated successfully."}

@router.post("/configure-whatsapp")
async def configure_whatsapp(config: WhatsAppConfig, db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(
            select(ClientConfig).where(ClientConfig.client_id == config.client_id)
        )
        existing = result.scalars().first()

        if existing:
            # Update fields directly, no need to add() again
            existing.phone_id = config.phone_id
            existing.token = config.token
            existing.verify_token = config.verify_token
        else:
            new_config = ClientConfig(
                client_id=config.client_id,
                phone_id=config.phone_id,
                token=config.token,
                verify_token=config.verify_token
            )
            db.add(new_config)

        await db.commit()
        return {"message": "WhatsApp config saved/updated successfully."}    
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=400, detail="Client ID already exists. It must be unique.")
    except SQLAlchemyError as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Database error: " + str(e.__class__.__name__))
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Unexpected error: " + str(e))

# @router.get("/webhook", tags=["Webhook Verification"])
# async def verify_webhook(
#     mode: str = Query(..., alias="hub.mode", description="Subscription mode"),
#     token: str = Query(..., alias="hub.verify_token", description="Verification token"),
#     challenge: str = Query(..., alias="hub.challenge", description="Challenge string"),
#     db: AsyncSession = Depends(get_db)
# ):
#     """
#     WhatsApp webhook verification endpoint.

#     - **mode**: Should be 'subscribe'
#     - **token**: Must match the verify_token stored in DB
#     - **challenge**: Returned as-is if verification passes
#     """
#     # Check if token exists in DB
#     result = await db.execute(
#         select(ClientConfig).where(ClientConfig.verify_token == token)
#     )
#     config = result.scalars().first()

#     if config:
#         return whatsapp_api.verify_webhook(mode, token, challenge)

#     raise HTTPException(status_code=403, detail="Invalid verify_token")


@router.get("/webhook", tags=["Webhook Verification"])  
async def verify_webhook(
    mode: str = Query(..., alias="hub.mode"),
    token: str = Query(..., alias="hub.verify_token"),
    challenge: str = Query(..., alias="hub.challenge"),
    db: AsyncSession = Depends(get_db)
):
    """
    WhatsApp webhook verification endpoint.
    """
    result = await db.execute(
        select(ClientConfig).where(ClientConfig.verify_token == token)
    )
    config = result.scalars().first()

    if config and mode == "subscribe":
        return int(challenge)  # WhatsApp expects the challenge to be echoed
    raise HTTPException(status_code=403, detail="Invalid verification")


@router.post("/webhook", tags=["Webhook Events"], summary="Receive WhatsApp messages", description="This endpoint handles POST requests from the WhatsApp webhook.")
async def webhook_listener(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Handles incoming WhatsApp messages/events and routes them to the proper client handler.
    """
    try:
        result = await db.execute(select(ClientConfig))
        configs = result.scalars().all()
        config_dict = {
            config.client_id: {
                "token": config.token,
                "phone_id": config.phone_id
            } for config in configs
        }

        return await whatsapp_api.handle_incoming_message(request, config_dict, db)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")


# @router.post("/send-message", tags=["WhatsApp Messaging"], summary="Send a WhatsApp message")
async def send_message(payload: SendMessagePayload, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ClientConfig).where(ClientConfig.client_id == payload.client_id)
    )
    config = result.scalars().first()

    if not config:
        raise HTTPException(status_code=404, detail="Client config not found")

    # ✅ Remove await if the function is not async
    return whatsapp_api.send_whatsapp_message_dynamic(
        token=config.token,
        phone_id=config.phone_id,
        to=payload.to,
        message=payload.message
    )

#tested 
@router.post("/send-template")
async def send_template_message(
    payload: TemplateMessagePayload,
    db: AsyncSession = Depends(get_db)
):  
    try:
        token, phone_id = await get_token_and_phone_id(payload.client_id, db)

        return await whatsapp_api.send_template_message(
            token=token,
            phone_id=phone_id,
            to=payload.to,
            template_name=payload.template_name,
            language_code=payload.language_code,
            components=payload.components
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

@router.post("/mark-as-read")
async def mark_as_read(payload: MarkReadPayload, db: AsyncSession = Depends(get_db)):
    config = await db.get(ClientConfig, payload.client_id)
    return await whatsapp_api.mark_message_as_read(config.token, payload.message_id)

# @router.get("/profile")
# async def get_business_profile(client_id: str, db: AsyncSession = Depends(get_db)):
#     ...
#     return await whatsapp_api.get_profile(token=ClientConfig.token)

# @router.post("/profile/update")
# async def update_profile(payload: UpdateProfilePayload, db: AsyncSession = Depends(get_db)):
#     ...
#     return await whatsapp_api.update_profile(token=ClientConfig.token, profile_data=payload.dict())

# @router.post("/send-media")
# async def send_media_message(payload: MediaMessagePayload, db: AsyncSession = Depends(get_db)):
#     ...
#     return await whatsapp_api.send_media_message(
#         token=config.token,
#         phone_id=config.phone_id,
#         to=payload.to,
#         media_id=payload.media_id,
#         media_type=payload.media_type,
#         caption=payload.caption
#     )
