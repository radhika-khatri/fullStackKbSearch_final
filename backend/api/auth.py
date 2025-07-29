from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from fastapi.responses import JSONResponse
from database.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from db import crud
from models import schemas
from supabase import create_client, Client

from utils.auth_utils import (
    authenticate_user,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    register_user_helper
)

router = APIRouter()

# Initialize Supabase client
supabase_url = "https://ddfuhlysbhdfjrnaadlk.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRkZnVobHlzYmhkZmpybmFhZGxrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMwODYyNjMsImV4cCI6MjA2ODY2MjI2M30.nTuVs5VNKd_MhYwrjslH1wr_XtOdCS0M3QnVEtcWp6g"
supabase: Client = create_client(supabase_url, supabase_key)

@router.post("/login", response_model=schemas.Token)
async def login(form_data: schemas.LoginRequest = Depends(), db: AsyncSession = Depends(get_db)):
    """
    Authenticate a user and return an access token if credentials are valid.
    Args:
        form_data (schemas.LoginRequest): The login request data containing email and password.
        db (AsyncSession): The async database session.
    Returns:
        dict: Access token and token type if authentication is successful.
    Raises:
        HTTPException: If authentication fails.
    """
    user = await authenticate_user(db, form_data.email, form_data.password, form_data.role)
    print(f"User authenticated:================> {user}")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials or role mismatch",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}
    return("success!!")


@router.post("/register", response_model=schemas.UserOut, status_code=201)
async def register_user(user: schemas.UserCreate, db: AsyncSession = Depends(get_db)):
    """
    Register a new user in the system.
    Args:
        user (schemas.UserCreate): The user creation data.
        db (AsyncSession): The async database session.
    Returns:
        schemas.UserOut: The created user object.
    """
    try:
        return await register_user_helper(db, user)
    except Exception as exc:
        # Log the exception for debugging
        print(f"Error in register_user: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during registration."
        )