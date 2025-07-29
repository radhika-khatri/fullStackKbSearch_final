from pymongo import MongoClient
from uuid import uuid4
from datetime import datetime

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['chat_db']
sessions = db['sessions']

# Generate a new chat session ID and create a session document
def create_chat_session(user_id):
    session_id = str(uuid4())
    session_doc = {
        'session_id': session_id,
        'user_id': user_id,
        'created_at': datetime.utcnow(),
        'messages': []
    }
    sessions.insert_one(session_doc)
    return session_id

# Store a message in a chat session
def store_message(session_id, sender, message):
    msg = {
        'sender': sender,
        'message': message,
        'timestamp': datetime.utcnow()
    }
    sessions.update_one(
        {'session_id': session_id},
        {'$push': {'messages': msg}}
    )

# Get chat history for a session
def get_chat_history(session_id):
    session = sessions.find_one({'session_id': session_id})
    if session:
        return session.get('messages', [])
    return []
