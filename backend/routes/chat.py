from fastapi import APIRouter, HTTPException, Body
from mongo.chat_sessions import create_chat_session, store_message, get_chat_history

router = APIRouter()

@router.post('/chat/session')
def create_session(user_id: str = Body(...)):
    session_id = create_chat_session(user_id)
    return {"session_id": session_id}

@router.post('/chat/message')
def add_message(session_id: str = Body(...), sender: str = Body(...), message: str = Body(...)):
    store_message(session_id, sender, message)
    return {"status": "success"}

@router.get('/chat/history/{session_id}')
def get_history(session_id: str):
    history = get_chat_history(session_id)
    return {"history": history}
