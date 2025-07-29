from pymongo import MongoClient
from datetime import datetime
import os
import redis
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory

# SQLite fallback import
try:
    from sqlite_fallback import save_feedback_sqlite, get_feedback_analytics_sqlite, save_chat_log_sqlite
    SQLITE_AVAILABLE = True
except ImportError:
    print("SQLite fallback not available")
    SQLITE_AVAILABLE = False

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://AppGallop:appgallop123@cluster0.tpatz5r.mongodb.net/")
print(f"Connecting to MongoDB with URI: {MONGO_URI[:50]}...")  # Log partial URI for debugging

MONGODB_AVAILABLE = False

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test the connection
    mongo_client.admin.command('ping')
    print("✅ MongoDB connection successful!")
    
    mongo_db = mongo_client["chat_support"]
    
    # Collections
    chat_logs_col = mongo_db["chat_logs"]
    ticket_logs_col = mongo_db["ticket_logs"]
    feedback_logs_col = mongo_db["feedback_logs"]
    
    MONGODB_AVAILABLE = True
    
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    if SQLITE_AVAILABLE:
        print("🔄 Falling back to SQLite database...")
    
    # Create dummy collections to prevent import errors
    mongo_client = None
    mongo_db = None
    chat_logs_col = None
    ticket_logs_col = None
    feedback_logs_col = None

# Redis setup (plain Redis)
redis_client = redis.Redis(host="localhost", port=6379, db=0)


def save_chat_log_mongo(**record):
    """
    Save a chat message with optional sentiment, embedding, and bot reply.
    Also updates Redis session context and LangChain memory.
    """

    # 1. Add timestamp
    record["timestamp"] = datetime.utcnow()

    # 2. Save to MongoDB
    chat_logs_col.insert_one(record)

    # 3. Update Redis context
    session_id = record.get("session_id")
    sender = record.get("sender")
    message = record.get("message")

    context = get_session_context(session_id)
    role_prefix = "User" if sender == "user" else "Bot"
    context.append(f"{role_prefix}: {message}")
    save_session_context(session_id, context)

    # 4. Update LangChain memory
    memory = get_langchain_memory(session_id)

    if sender == "user":
        memory.save_context({"input": message}, {"output": record.get("bot_reply", "")})

    return record  # Optionally return record or inserted ID


def save_ticket_log_mongo(ticket_data):
    """
    Save a support ticket.
    """
    ticket_logs_col.insert_one({
        **ticket_data,
        "created_at": datetime.utcnow()
    })


def save_feedback_mongo(message_id, feedback, session_id=None, comment=None):
    """
    Save thumbs up/down feedback.
    Uses MongoDB if available, otherwise falls back to SQLite.
    """
    if MONGODB_AVAILABLE and feedback_logs_col is not None:
        try:
            result = feedback_logs_col.insert_one({
                "message_id": message_id,
                "feedback": feedback,  # 'up' or 'down'
                "session_id": session_id,
                "comment": comment,
                "timestamp": datetime.utcnow()
            })
            print(f"✅ Feedback saved to MongoDB: {message_id} - {feedback}")
            return result
        except Exception as e:
            print(f"❌ Failed to save feedback to MongoDB: {e}")
            if SQLITE_AVAILABLE:
                print("🔄 Falling back to SQLite...")
                return save_feedback_sqlite(message_id, feedback, session_id, comment)
            raise e
    elif SQLITE_AVAILABLE:
        print(f"📝 Using SQLite for feedback: {message_id} - {feedback}")
        return save_feedback_sqlite(message_id, feedback, session_id, comment)
    else:
        raise Exception("Neither MongoDB nor SQLite is available for feedback storage")


def get_session_context(session_id):
    """
    Load recent session context from Redis (manual).
    Returns a list of strings.
    """
    redis_key = f"session:{session_id}"
    if redis_client.exists(redis_key):
        return json.loads(redis_client.get(redis_key))
    return []


def save_session_context(session_id, context_list):
    """
    Save updated context back to Redis (manual).
    """
    redis_key = f"session:{session_id}"
    redis_client.set(redis_key, json.dumps(context_list))


def get_langchain_memory(session_id):
    """
    Returns a LangChain ConversationBufferMemory object backed by Redis.
    This automatically handles message appending, retrieval, and formatting.
    """
    # Create Redis-backed chat history
    chat_history = RedisChatMessageHistory(
        url="redis://localhost:6379",
        session_id=session_id
    )
    
    # Wrap in LangChain memory
    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        return_messages=False
    )
    return memory
