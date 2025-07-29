from sqlalchemy.orm import Session
from models.db_models import ChatLog, Ticket, UserFeedback, HumanEscalation
from models.schemas import ChatMessage, TicketCreate, FeedbackCreate, EscalationCreate
from sqlalchemy.future import select
from database.database import async_session
from models.db_models import User
from mongo.models import save_chat_log_mongo, save_ticket_log_mongo, save_feedback_mongo, chat_logs_col
from sqlalchemy.ext.asyncio import AsyncSession
# 💬 Save Chat Message
from datetime import datetime
from bson import ObjectId


def save_chat_message(user_id, session_id, message, sender, sentiment=None, embedding=None, bot_reply=None):
    chat_data = {
        "user_id": user_id,
        "session_id": session_id,
        "message": message,
        "sender": sender,
        "sentiment": sentiment,
        "embedding": embedding,
        "bot_reply": bot_reply
    }
    save_chat_log_mongo(**chat_data)

# 📩 Get All Chat Logs for a User
def get_user_chats(db: Session, user_id: int):
    return db.query(ChatLog).filter(ChatLog.user_id == user_id).order_by(ChatLog.timestamp).all()

#from bson import ObjectId

def serialize_mongo_doc(doc):
    return {
        "message": doc.get("message"),
        "bot_reply": doc.get("bot_reply")
    }

def get_conversation_context(session_id: str, limit: int = 10):
    print("Session ID:", session_id)
    results = list(chat_logs_col.find(
        {"session_id": session_id}
    ).sort("timestamp", -1).limit(limit))[::-1]  # Oldest to newest

    serialized_results = [serialize_mongo_doc(doc) for doc in results]
    print(serialized_results)

    return serialized_results

# 🎟️ Create Support Ticket
def create_ticket(ticket_data):
    return save_ticket_log_mongo(ticket_data)

# 🎟️ Get Ticket by ID
def get_ticket(db: Session, ticket_id: int):
    return db.query(Ticket).filter(Ticket.id == ticket_id).first()

# 🎟️ Update Ticket Status
def update_ticket_status(db: Session, ticket_id: int, status: str):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if ticket:
        ticket.status = status
        db.commit()
        db.refresh(ticket)
    return ticket

# 🎟️ Delete Ticket
def delete_ticket(db: Session, ticket_id: int):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if ticket:
        db.delete(ticket)
        db.commit()
    return ticket

# 👍👎 Save Feedback
def save_feedback(message_id: str, feedback: str, session_id: str = None, comment: str = None):
    return save_feedback_mongo(message_id, feedback, session_id, comment)

#  Get Feedback by Chat ID
def get_feedback_for_chat(db: Session, chat_id: int):
    return db.query(UserFeedback).filter(UserFeedback.chat_id == chat_id).all()

#  Save Human Escalation Record
def escalate_to_human(db: Session, escalation: EscalationCreate):
    es = HumanEscalation(**escalation.dict())
    db.add(es)
    db.commit()
    db.refresh(es)
    return es

# 📥 Get All Escalations
def get_escalation_queue(db: Session):
    return db.query(HumanEscalation).order_by(HumanEscalation.timestamp.desc()).all()

# async def get_user_by_email(email: str):
#     async with async_session() as session:
#         result = await session.execute(select(User).where(User.email == email))
#         user = result.scalar_one_or_none()
#         return user


async def get_user_by_email(db: AsyncSession, email: str):
    """
    Retrieve a user from the database by their email address using an async session.
    Args:
        db (AsyncSession): The async database session.
        email (str): The user's email address.
    Returns:
        User or None: The user object if found, else None.
    """
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    return user

async def create_user(db: AsyncSession, user, hashed_password: str):
    """
    Create a new user in the database using an async session.
    Args:
        db (AsyncSession): The async database session.
        user: The user data object with attributes name, email, and role.
        hashed_password (str): The hashed password for the user.
    Returns:
        User: The newly created user object.
    """
    db_user = User(
        name=user.name,
        email=user.email,
        hashed_password=hashed_password,
        role=user.role,  # <-- added role
        is_active=True
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

    


