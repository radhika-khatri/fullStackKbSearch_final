from typing import Text
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Text, DateTime,func
from sqlalchemy.orm import relationship
from database.database import Base
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID
import uuid


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    role = Column(String, default="user")  # <-- add this line
    file_id = Column(String, nullable=True)

    # these two must correspond to the other side’s back_populates
    items = relationship("Item", back_populates="owner")
    tickets = relationship("Ticket", back_populates="user")
    feedbacks = relationship(
        "UserFeedback",
        back_populates="user",
        cascade="all, delete-orphan"
    )


class Item(Base):
    __tablename__ = "items" 

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), nullable=False)
    description = Column(String, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="items")

class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    user_input = Column(Text)
    bot_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Ticket(Base):    
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.email"))
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(50), default="open")
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="tickets")   

class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.email"))
    feedback_text = Column(Text, nullable=False)
    rating = Column(Integer, nullable=False)  # e.g., 1-5 scale
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="feedbacks")


class HumanEscalation(Base):
    __tablename__ = "human_escalations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    reason = Column(Text)
    escalated_at = Column(DateTime, default=datetime.utcnow)

class ClientConfig(Base):
    __tablename__ = "client_config"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(String, nullable=False)
    sender = Column(String, nullable=False)  # e.g., 'user' or 'bot'
    user_number = Column(String, nullable=False)  # WhatsApp number
    message = Column(Text, nullable=False)
    message_type = Column(String, default="text")  # Optional: 'text', 'image', etc.
    direction = Column(String, nullable=False)  # 'incoming' or 'outgoing'
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(String, nullable=False)
    sender = Column(String, nullable=False)  # e.g., 'user' or 'bot'
    user_number = Column(String, nullable=False)  # WhatsApp number
    message = Column(Text, nullable=False)
    message_type = Column(String, default="text")  # Optional: 'text', 'image', etc.
    direction = Column(String, nullable=False)  # 'incoming' or 'outgoing'
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class ClassifyLabels(Base):
    __tablename__ = "classify_labels"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    label = Column(Text)

class DeviceInfo(Base):
    __tablename__ = "device_info"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    ip = Column(String, index=True)
    os = Column(String)
    browser = Column(String)
    user_agent = Column(String)
    screen_resolution = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)