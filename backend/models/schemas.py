# This file contains Pydantic schemas for the application.

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Union
from datetime import datetime
from enum import Enum  


# ----------------------------- USER SCHEMAS ----------------------------- #

class UserBase(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str  # <-- add this
    file_id: str

class RoleEnum(str, Enum):
    user = "user"
    admin = "admin"

class LoginRequest(BaseModel):
    email: str
    password: str
    role: RoleEnum


# class UserCreate(UserBase):
#     password: str


class UserResponse(UserBase):
    id: int
    is_active: bool

    class Config:
        # orm_mode = True
        from_attributes = True


# ----------------------------- ITEM SCHEMAS ----------------------------- #

class ItemBase(BaseModel):
    title: str
    description: Optional[str] = None


class ItemCreate(ItemBase):
    pass


class ItemResponse(ItemBase):
    id: int
    owner_id: int

    class Config:
        # orm_mode = True
        from_attributes = True


# -------------------------- AUTH & TOKEN SCHEMAS ------------------------ #

class Token(BaseModel):
    access_token: str
    token_type: str
    role: Optional[str] = None
    user: Optional[dict] = None


class TokenData(BaseModel):
    email: Optional[str] = None


# --------------------------- CHAT MESSAGE SCHEMAS ----------------------- #

class ChatMessage(BaseModel):
    user_id: Optional[int]
    message: str
    sender: str  # 'user' or 'bot'
    timestamp: datetime | None = None
    session_id: str

    class Config:
        # orm_mode = True
        from_attributes = True


# ---------------------------- TICKET SCHEMAS ---------------------------- #

class TicketBase(BaseModel):
    subject: str
    user_id: int
    priority: Optional[str] = "normal"
    status: Optional[str] = "open"
    created_at: Optional[datetime] = None


class TicketCreate(TicketBase):
    description: str


class TicketResponse(TicketBase):
    id: int
    description: str

    class Config:
        # orm_mode = True
        from_attributes = True


# -------------------------- FEEDBACK SCHEMAS ---------------------------- #

class FeedbackCreate(BaseModel):
    ticket_id: int
    rating: int  # 1 to 5
    comment: Optional[str] = None
    sentiment: Optional[str] = None

    class Config:
        # orm_mode = True
        from_attributes = True


# -------------------------- ESCALATION SCHEMAS -------------------------- #

class EscalationCreate(BaseModel):
    ticket_id: int
    reason: str
    escalated_to: str
    timestamp: Optional[datetime] = None

    class Config:
        # orm_mode = True
        from_attributes = True


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: RoleEnum = Field(default=RoleEnum.user)

class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    is_active: bool
    role: Optional[str] = None

    class Config:
        # orm_mode = True
        from_attributes = True


class AskRequest(BaseModel):
    question: str
    file_id: str

class AskResponse(BaseModel):
    answer: str

class LabelRequest(BaseModel):
    label: str
