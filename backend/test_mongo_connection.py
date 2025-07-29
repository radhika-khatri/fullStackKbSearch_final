#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime

# Load environment variables
load_dotenv()

def test_mongodb_connection():
    try:
        print("🔄 Testing MongoDB connection...")
        
        # Get MongoDB URI
        MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://AppGallop:appgallop123@cluster0.tpatz5r.mongodb.net/chat_support?retryWrites=true&w=majority")
        print(f"📡 Using URI: {MONGO_URI[:50]}...")
        
        # Create client with timeout
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
        
        # Test connection
        print("🏓 Pinging MongoDB...")
        result = client.admin.command('ping')
        print(f"✅ Ping successful: {result}")
        
        # Test database access
        db = client["chat_support"]
        print(f"📊 Database: {db.name}")
        
        # Test collection access
        feedback_col = db["feedback_logs"]
        print(f"📁 Collection: {feedback_col.name}")
        
        # Test insert
        print("💾 Testing insert...")
        test_doc = {
            "message_id": "test_123",
            "feedback": "up",
            "timestamp": datetime.utcnow(),
            "test": True
        }
        
        result = feedback_col.insert_one(test_doc)
        print(f"✅ Insert successful: {result.inserted_id}")
        
        # Test count
        count = feedback_col.count_documents({})
        print(f"📊 Total documents: {count}")
        
        # Clean up test document
        feedback_col.delete_one({"_id": result.inserted_id})
        print("🗑️ Test document cleaned up")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_mongodb_connection()
