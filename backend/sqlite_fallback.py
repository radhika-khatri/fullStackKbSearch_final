import sqlite3
import os
from datetime import datetime
import json

# SQLite database path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'feedback.db')

def init_sqlite_db():
    """Initialize SQLite database for feedback storage"""
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT NOT NULL,
            feedback TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            comment TEXT
        )
    ''')
    
    # Create chat logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            session_id TEXT NOT NULL,
            message TEXT NOT NULL,
            sender TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            sentiment TEXT,
            embedding TEXT,
            bot_reply TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"SQLite database initialized at: {DB_PATH}")

def save_feedback_sqlite(message_id, feedback, session_id=None, comment=None):
    """Save feedback to SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback_logs (message_id, feedback, session_id, comment, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (message_id, feedback, session_id, comment, datetime.utcnow().isoformat()))
        
        conn.commit()
        feedback_id = cursor.lastrowid
        conn.close()
        
        print(f"Feedback saved to SQLite: {message_id} - {feedback}")
        return {"feedback_id": feedback_id, "message_id": message_id}
        
    except Exception as e:
        print(f"Failed to save feedback to SQLite: {e}")
        raise e

def get_feedback_analytics_sqlite():
    """Get feedback analytics from SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Count total feedback
        cursor.execute("SELECT COUNT(*) FROM feedback_logs")
        total_feedback = cursor.fetchone()[0]
        
        # Count positive feedback
        cursor.execute("SELECT COUNT(*) FROM feedback_logs WHERE feedback = 'up'")
        positive_feedback = cursor.fetchone()[0]
        
        # Count negative feedback
        cursor.execute("SELECT COUNT(*) FROM feedback_logs WHERE feedback = 'down'")
        negative_feedback = cursor.fetchone()[0]
        
        # Calculate satisfaction rate
        satisfaction_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
        
        conn.close()
        
        return {
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "satisfaction_rate": round(satisfaction_rate, 2)
        }
        
    except Exception as e:
        print(f"Failed to get analytics from SQLite: {e}")
        return {
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "satisfaction_rate": 0
        }

def save_chat_log_sqlite(user_id, session_id, message, sender, sentiment=None, embedding=None, bot_reply=None):
    """Save chat log to SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Convert embedding to JSON string if it exists
        embedding_str = json.dumps(embedding) if embedding else None
        
        cursor.execute('''
            INSERT INTO chat_logs (user_id, session_id, message, sender, sentiment, embedding, bot_reply, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, session_id, message, sender, sentiment, embedding_str, bot_reply, datetime.utcnow().isoformat()))
        
        conn.commit()
        log_id = cursor.lastrowid
        conn.close()
        
        return {"log_id": log_id, "session_id": session_id}
        
    except Exception as e:
        print(f"Failed to save chat log to SQLite: {e}")
        raise e

# Initialize the database when module is imported
init_sqlite_db()
