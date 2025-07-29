import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mongo.chat_sessions import create_chat_session, store_message, get_chat_history

# Test user id
test_user_id = 'user123'

# 1. Generate chat session ID
session_id = create_chat_session(test_user_id)
print('Session ID:', session_id)

# 2. Store chat history
store_message(session_id, 'user', 'Hello!')
store_message(session_id, 'bot', 'Hi, how can I help you?')
store_message(session_id, 'user', 'Tell me about AppG-AI.')
store_message(session_id, 'bot', 'AppG-AI is your intelligent business platform.')

# 3. Retrieve chat history
history = get_chat_history(session_id)
print('Chat History:')
for msg in history:
    print(msg)
