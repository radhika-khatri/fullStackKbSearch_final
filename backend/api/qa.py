# api/qa.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from models.schemas import AskRequest, AskResponse
from utils.embed_store import load_index, embed_question, query_embeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import re
import time
import asyncio
import json
import random
from difflib import SequenceMatcher
from PIL import Image
import pytesseract
import base64
import io
import requests
import logging

# Import Gemini Vision processing
from services.gemini_vision import process_image_with_gemini_and_ocr
import google.generativeai as genai

# Configure Gemini for text chat as well
GEMINI_API_KEY = "AIzaSyB5V3qgB25MFkv79JGaHUH75G047iQ5VIU"
genai.configure(api_key=GEMINI_API_KEY)

# Setup logging
logger = logging.getLogger(__name__)

from utils.embed_store import VECTOR_DIR

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM_MODEL = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

# Comprehensive conversation patterns with responses
CONVERSATION_PATTERNS = {
    # Greetings and casual interactions
    "greetings": {
        "patterns": ["hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening", "good night", "howdy", "what's up", "sup", "yo", "hiya", "salutations"],
        "responses": [
            "Hello! 👋 I'm your AppGallop AI assistant. How can I help you today?",
            "Hi there! 😊 Great to see you! What can I do for you?",
            "Hey! Welcome to AppGallop! I'm here to help with any questions you have.",
            "Hello! Hope you're having a wonderful day! How may I assist you?",
            "Hi! I'm your friendly AI assistant. What would you like to know?"
        ]
    },
    
    # How are you variations
    "wellbeing": {
        "patterns": ["how are you", "how are you doing", "how's it going", "how do you feel", "are you okay", "are you well", "how have you been", "what's your status", "how are things"],
        "responses": [
            "I'm doing great, thank you for asking! I'm here and ready to assist you. What would you like to know?",
            "I'm fantastic! Always ready to help. How are you doing today?",
            "I'm doing wonderful! Thanks for asking. How can I help you today?",
            "I'm excellent and fully operational! What can I help you with?",
            "I'm doing amazing! Hope you're having a great day too. What's on your mind?"
        ]
    },
    
    # Capabilities and help
    "capabilities": {
        "patterns": ["what can you do", "what are your capabilities", "help", "help me", "what do you know", "what are you good at", "what's your purpose", "how can you help", "what services do you offer"],
        "responses": [
            "I can help you with various tasks! I can answer questions, provide information, assist with problem-solving, and have conversations. Feel free to upload a knowledge base for more specific domain expertise. What would you like help with?",
            "I'm here to assist with answering questions, providing information, helping solve problems, and having meaningful conversations. I work best when you upload knowledge bases for specific topics!",
            "I can help with questions, provide support, assist with research, and chat about various topics. Upload a knowledge base for specialized help in your domain!",
            "My capabilities include answering questions, providing assistance, problem-solving, and engaging conversations. I become even more powerful with uploaded knowledge bases!"
        ]
    },
    
    # Identity questions
    "identity": {
        "patterns": ["who are you", "what are you", "tell me about yourself", "introduce yourself", "what's your name", "are you human", "are you a bot", "are you ai", "are you real"],
        "responses": [
            "I'm AppGallop's AI assistant! I'm designed to help answer questions, provide support, and assist with various tasks. I can work with uploaded knowledge bases to give you specific information about your business or topics of interest.",
            "I'm your friendly AI assistant from AppGallop! I'm here to help with questions, provide information, and make your experience smoother.",
            "I'm an AI assistant created by AppGallop to help users like you! I can answer questions, provide assistance, and work with knowledge bases for specialized support.",
            "I'm AppGallop's intelligent assistant! I'm designed to be helpful, informative, and friendly while assisting with your queries and tasks."
        ]
    },
    
    # Gratitude
    "gratitude": {
        "patterns": ["thank you", "thanks", "thank you so much", "appreciate it", "thanks a lot", "much appreciated", "grateful", "cheers", "thx", "ty"],
        "responses": [
            "You're very welcome! I'm happy to help. Is there anything else you'd like to know or discuss?",
            "My pleasure! Always glad to assist. What else can I help you with?",
            "You're welcome! I'm here whenever you need help. Anything else on your mind?",
            "Happy to help! Feel free to ask if you have more questions.",
            "You're most welcome! I'm always here to assist. What else would you like to know?"
        ]
    },
    
    # Farewells
    "farewells": {
        "patterns": ["bye", "goodbye", "see you later", "farewell", "see ya", "catch you later", "until next time", "take care", "have a good day", "talk to you later"],
        "responses": [
            "Goodbye! It was great chatting with you. Feel free to come back anytime if you need help! 👋",
            "See you later! Thanks for the great conversation. Come back anytime!",
            "Farewell! Hope to chat with you again soon. Take care! 😊",
            "Bye! It was wonderful helping you today. Don't hesitate to return if you need assistance!",
            "Take care! Thanks for visiting. I'll be here whenever you need help! 👋"
        ]
    },
    
    # Compliments and praise
    "compliments": {
        "patterns": ["you're great", "you're awesome", "good job", "well done", "you're helpful", "you're smart", "you're amazing", "love you", "you rock", "fantastic work"],
        "responses": [
            "Thank you so much! That really makes my day. I'm here to help whenever you need it! 😊",
            "Aww, thank you! I really appreciate that. Is there anything else I can help you with?",
            "You're too kind! I'm just doing what I love - helping people like you!",
            "Thank you for the kind words! It motivates me to keep being helpful. What else can I do for you?",
            "That's so nice of you to say! I'm glad I could be helpful. Anything else you'd like to discuss?"
        ]
    },
    
    # Questions about time, weather, location
    "external_info": {
        "patterns": ["what time is it", "what's the weather", "where am i", "what's the date", "what day is it", "current time", "today's weather", "temperature outside"],
        "responses": [
            "I don't have access to real-time information like current time or weather, but you can check your device for that info! Is there something else I can help you with?",
            "I wish I could tell you the current time/weather, but I don't have access to real-time data. Check your device for that! What else can I assist with?",
            "I don't have access to live data like time or weather updates. Your device can give you that info! How else can I help you today?"
        ]
    },
    
    # Fun and casual questions
    "fun_questions": {
        "patterns": ["tell me a joke", "make me laugh", "something funny", "entertain me", "tell me something interesting", "surprise me", "what's new", "anything exciting"],
        "responses": [
            "Here's one for you: Why don't scientists trust atoms? Because they make up everything! 😄 What else can I help you with?",
            "How about this: I told my computer a joke about UDP, but it didn't get it! 😂 Need help with anything else?",
            "Fun fact: The word 'set' has the most different meanings in English! Pretty interesting, right? What would you like to know more about?",
            "Here's something cool: Honey never spoils! Archaeologists have found edible honey in ancient Egyptian tombs. Amazing, isn't it? What else interests you?",
            "Did you know? A group of flamingos is called a 'flamboyance'! 🦩 What other topics would you like to explore?"
        ]
    },
    
    # Emotional support
    "emotional_support": {
        "patterns": ["i'm sad", "i'm tired", "i'm stressed", "i'm worried", "i'm anxious", "i'm confused", "i'm frustrated", "i need help", "i'm lost", "i don't know"],
        "responses": [
            "I'm sorry you're feeling that way. I'm here to help however I can. Would you like to talk about what's bothering you?",
            "That sounds tough. I'm here to listen and help in any way possible. What's on your mind?",
            "I understand that can be difficult. I'm here to support you. Would you like to share what's causing these feelings?",
            "I'm here for you. Sometimes talking things through can help. What would you like to discuss?",
            "That must be challenging. I'm here to help and support you through this. How can I assist you today?"
        ]
    },
    
    # Learning and education
    "learning": {
        "patterns": ["teach me", "i want to learn", "explain", "how does this work", "i don't understand", "can you show me", "educate me", "i'm curious about"],
        "responses": [
            "I'd love to help you learn! What topic are you interested in? I can explain things or point you in the right direction.",
            "Learning is fantastic! What would you like to know more about? I'm here to help explain things clearly.",
            "Great attitude! I'm here to help you understand. What subject or topic interests you?",
            "I'm excited to help you learn! What area would you like to explore? I'll do my best to explain it clearly.",
            "Learning new things is wonderful! What topic has caught your curiosity? I'm here to help!"
        ]
    },
    
    # Work and productivity
    "work_productivity": {
        "patterns": ["i'm working on", "need help with work", "productivity tips", "how to be more efficient", "work advice", "career guidance", "job help"],
        "responses": [
            "I'd be happy to help with work-related questions! What specific area are you working on? I can provide guidance and suggestions.",
            "Work can be challenging! I'm here to help with productivity tips, problem-solving, or any work-related questions you have.",
            "Great that you're focused on work! What aspect would you like help with? I can offer advice and support.",
            "I'm here to help with work matters! Whether it's productivity, problem-solving, or guidance, what do you need assistance with?"
        ]
    },
    
    # Technology questions
    "technology": {
        "patterns": ["computer problem", "tech help", "software issue", "app not working", "technical support", "it problem", "bug", "error", "crash"],
        "responses": [
            "I can try to help with tech issues! What specific problem are you experiencing? Describe what's happening and I'll do my best to assist.",
            "Tech problems can be frustrating! Tell me more about what's going wrong and I'll help troubleshoot the issue.",
            "I'm here to help with technical issues! What device or software is giving you trouble? Let's work through it together.",
            "Technical support is one of my strengths! Describe the problem you're facing and I'll guide you through potential solutions."
        ]
    },
    
    # Random conversation starters
    "random_conversation": {
        "patterns": ["i'm bored", "let's chat", "talk to me", "i'm lonely", "keep me company", "what should we discuss", "random question", "tell me something"],
        "responses": [
            "I'm here to chat! What's on your mind today? We could talk about hobbies, interests, current events, or anything else you'd like!",
            "I'd love to keep you company! What would you like to talk about? I'm interested in hearing about your day or any topics you enjoy.",
            "Great! I enjoy good conversations. What interests you? We could discuss books, movies, science, travel, or whatever you're passionate about!",
            "I'm happy to chat with you! What's something that's been on your mind lately? Or is there a topic you'd like to explore together?",
            "Perfect timing for a chat! What would make for an interesting conversation today? I'm all ears!"
        ]
    }
}

def similarity_score(a, b):
    """Calculate similarity between two strings using SequenceMatcher"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_best_match(user_input, patterns, threshold=0.6):
    """Find the best matching pattern for user input with similarity threshold"""
    user_input = user_input.lower().strip()
    best_match = None
    best_score = 0
    
    for category, data in patterns.items():
        for pattern in data["patterns"]:
            # Direct substring match (higher priority)
            if pattern in user_input or user_input in pattern:
                score = 0.9
            else:
                # Fuzzy matching
                score = similarity_score(user_input, pattern)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = category
    
    return best_match, best_score

def get_gemini_chat_response(question: str) -> str:
    """Use Gemini AI for intelligent chat responses when KB/patterns don't match well"""
    try:
        logger.info("Using Gemini for intelligent chat response...")
        
        # Initialize Gemini model for text
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create a conversational prompt
        prompt = f"""
You are AppGallop's helpful AI assistant. A user asked: "{question}"

Please provide a friendly, conversational response that:
1. Directly addresses their question in a natural, helpful way
2. Keeps the tone warm and professional 
3. Offers to help further if needed
4. Stays concise (2-3 sentences max)
5. If you don't know something specific, admit it but offer general help

Respond as a knowledgeable, friendly assistant would in a conversation.
"""
        
        # Generate response
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "I'd be happy to help you with that! Could you tell me a bit more about what you're looking for?"
            
    except Exception as e:
        logger.error(f"Gemini chat response failed: {e}")
        return "I'm here to help! While I'm having some technical difficulties right now, feel free to ask me anything and I'll do my best to assist you."

import random

@router.post("/ask", response_model=AskResponse)
async def ask_question(data: AskRequest):
    file_id = data.file_id
    question = data.question

    vector_path = os.path.join(VECTOR_DIR, f"{file_id}.index")
    if not os.path.exists(vector_path):
        raise HTTPException(status_code=404, detail="Vector index not found for this file")

    index, chunks = load_index(file_id)
    q_embedding = embed_question(question)
    D, I = index.search(q_embedding, k=3)

    retrieved = [chunks[i] for i in I[0]]
    joined_context = "\n".join(retrieved)
    prompt = f"Context:\n{joined_context}\n\nUser: {question}\nAI:"

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    clean_answer = answer.split("AI:")[-1].lstrip().strip()
    # return AskResponse(answer=answer.split("AI:")[-1].strip())
    return AskResponse(answer=clean_answer)


@router.post("/v1/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        top_chunks = query_embeddings(request.file_id, request.question, top_k=5)
        context = "\n".join(top_chunks)
        prompt = f"Context:\n{context}\n\nUser: {request.question}\n\nAI:"

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        clean_answer = answer.split("AI:")[-1].lstrip().strip()
        return {"answer": clean_answer}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed: {str(e)}")


@router.post("/static-chat")
async def static_chat(request: dict):
    """Advanced conversational AI with knowledge base search and fuzzy matching"""
    try:
        question = request.get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        # First, check for basic conversational patterns (high priority for greetings, etc.)
        best_match, confidence = find_best_match(question, CONVERSATION_PATTERNS)
        
        # Handle basic conversational patterns first (with higher confidence threshold)
        if best_match and confidence > 0.7:
            # These are clearly conversational - respond immediately
            if best_match in ['greetings', 'wellbeing', 'gratitude', 'farewells', 'compliments', 'fun_questions']:
                responses = CONVERSATION_PATTERNS[best_match]["responses"]
                response = random.choice(responses)
                return {"answer": response}
        
        # For business/knowledge queries, search the knowledge base first
        try:
            # 1. Try JSON file similarity search first (more accurate for exact questions)
            # Load knowledge base entries from uploaded JSON files
            kb_entries = []
            from pathlib import Path
            import json
            
            upload_dir = Path("uploads")
            upload_files = list(upload_dir.glob("*.json"))
            
            for file_path in upload_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                                    kb_entries.append({
                                        "question": item['question'],
                                        "answer": item['answer'],
                                        "category": item.get('category', 'General')
                                    })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
            
            # Search for the best matching knowledge base entry
            if kb_entries:
                best_match_entry = None
                best_similarity = 0
                
                for entry in kb_entries:
                    # Calculate similarity with the question
                    question_similarity = similarity_score(question.lower(), entry['question'].lower())
                    
                    # Boost score for exact key phrase matches
                    question_words = question.lower().split()
                    entry_question_words = entry['question'].lower().split()
                    
                    # Check for important keyword matches
                    important_keywords = ['time', 'long', 'develop', 'cost', 'price', 'services', 'offer', 'technologies', 'what', 'how']
                    keyword_boost = 0
                    for word in question_words:
                        if word in important_keywords and word in entry_question_words:
                            keyword_boost += 0.2
                    
                    # Also check similarity with answer keywords (but lower weight)
                    answer_similarity = similarity_score(question.lower(), entry['answer'].lower()) * 0.5
                    
                    max_similarity = max(question_similarity + keyword_boost, answer_similarity)
                    
                    if max_similarity > best_similarity:
                        best_similarity = max_similarity
                        best_match_entry = entry
                
                # If we found a good match (similarity > 0.4), return it
                if best_match_entry and best_similarity > 0.4:
                    logger.info(f"Found KB match with similarity {best_similarity:.2f}")
                    return {"answer": best_match_entry['answer']}
            
            # 2. Try vector embedding search as fallback (for more complex queries)
            try:
                from utils.embed_store import query_embeddings
                # Look for vector indexes in the vector_stores directory
                import os
                vector_dir = "vector_stores"
                if os.path.exists(vector_dir):
                    index_files = [f for f in os.listdir(vector_dir) if f.endswith('.index')]
                    if index_files:
                        # Use the first available index (or you could search all)
                        file_id = index_files[0].replace('.index', '')
                        relevant_chunks = query_embeddings(file_id, question, top_k=3)
                        if relevant_chunks:
                            # Extract clean answers from the chunks
                            clean_answers = []
                            for chunk in relevant_chunks[:2]:  # Take top 2 chunks
                                # Extract answer part from "Q: ... A: ..." format
                                if "A:" in chunk:
                                    answer_part = chunk.split("A:")[-1].strip()
                                    if answer_part and len(answer_part) > 20:
                                        clean_answers.append(answer_part)
                                elif len(chunk) > 20:  # If no Q&A format, use the chunk directly
                                    clean_answers.append(chunk.strip())
                            
                            if clean_answers:
                                # Return the most relevant answer, or combine if multiple
                                if len(clean_answers) == 1:
                                    logger.info("Found vector search match")
                                    return {"answer": clean_answers[0]}
                                else:
                                    # Combine answers intelligently
                                    combined = clean_answers[0]
                                    if len(clean_answers) > 1 and not clean_answers[0].endswith('.'):
                                        combined += ". " + clean_answers[1]
                                    logger.info("Found combined vector search match")
                                    return {"answer": combined}
            except Exception as e:
                print(f"Vector search error: {e}")
            
        except Exception as e:
            print(f"Knowledge base search error: {e}")
            # Continue to conversational patterns if KB search fails
        
        # Now try conversational patterns (for questions that didn't match KB well)
        if best_match and confidence > 0.5:
            # Get a random response from the matched category
            responses = CONVERSATION_PATTERNS[best_match]["responses"]
            response = random.choice(responses)
            logger.info(f"Using conversation pattern: {best_match} (confidence: {confidence:.2f})")
            return {"answer": response}
        
        # Handle specific keywords with fuzzy matching for edge cases
        question_lower = question.lower()
        
        # Business/AppGallop related questions that didn't match KB
        if any(word in question_lower for word in ["appgallop", "company", "business"]) and "what is" not in question_lower:
            return {"answer": "I'd be happy to help you with AppGallop-related questions! For specific details about our services, pricing, or processes, please feel free to ask more specific questions. What would you like to know?"}
        
        # Programming/Development questions that didn't match KB
        if any(word in question_lower for word in ["code", "programming", "development", "api", "software", "bug", "debug"]):
            return {"answer": "I can help with programming and development questions! While I'd need a relevant knowledge base for specific technical details, I'm happy to discuss general concepts. What are you working on?"}
        
        # Very general questions
        if any(word in question_lower for word in ["what", "how", "why", "when", "where"]) and len(question.split()) <= 3:
            return {"answer": f"That's a broad question! Could you be more specific about what aspect of '{question}' you'd like to know about? I'm here to help with both general conversation and specific topics if you have a knowledge base uploaded."}
        
        # 🚀 NEW: Use Gemini AI as intelligent fallback for unmatched questions
        try:
            logger.info("No good KB/pattern match found, using Gemini AI for intelligent response...")
            gemini_response = get_gemini_chat_response(question)
            return {"answer": gemini_response}
        except Exception as e:
            logger.error(f"Gemini fallback failed: {e}")
            # Final fallback to default response
            return {"answer": f"I'd love to help you with that! While I can chat about general topics, I work best when you upload a knowledge base with specific information. For now, feel free to ask me anything - whether it's a casual conversation or if you'd like to know more about AppGallop's services. What's on your mind?"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed: {str(e)}")


# Image processing helper functions
def extract_text_from_image_simple(image: Image.Image) -> str:
    """Fast text extraction from image using OCR"""
    try:
        if not image:
            return "No text detected in the image."
            
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (faster processing)
        width, height = image.size
        if width > 1200 or height > 1200:
            ratio = min(1200/width, 1200/height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Fast OCR with single config
        config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=config)
        
        if not text.strip():
            return "No text detected in the image."
        
        # Quick cleanup
        text = ' '.join(text.split())
        return text
        
    except Exception as e:
        error_msg = f"OCR extraction failed: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        print(f"DEBUG: {error_msg}")  # Also print for debugging
        return "Failed to extract text from image."
        return "Failed to extract text from image."

def image_to_base64_simple(image: Image.Image) -> str:
    """Fast conversion of PIL Image to base64 string"""
    try:
        if not image:
            raise ValueError("Image is None or empty")
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Quick compression for speed
        width, height = image.size
        if width > 800 or height > 800:
            ratio = min(800/width, 800/height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=60, optimize=True)  # Lower quality for speed
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        if not img_base64:
            raise ValueError("Base64 conversion resulted in empty string")
            
        return img_base64
        
    except Exception as e:
        error_msg = f"Base64 conversion failed: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        print(f"DEBUG: {error_msg}")  # Also print for debugging
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

async def query_ollama_for_chat(base64_image: str, question: str, ocr_text: str) -> str:
    """Fast query to local Ollama LLaVA model for chat"""
    try:
        # Simple, focused prompt optimized for LLaVA
        if question and question.strip():
            prompt = question.strip()[:150]  # Shorter prompt for LLaVA
        else:
            prompt = "What do you see?"  # Very simple default
        
        # Don't include OCR text in prompt for faster processing
        # LLaVA can see text in images directly
        
        payload = {
            "model": "llava:latest",
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Lower for consistency
                "top_p": 0.9,
                "num_predict": 100   # Shorter responses for speed
            }
        }
        
        # Optimized timeout for LLaVA
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=45  # Reasonable timeout for LLaVA
        )
        
        if response.status_code == 200:
            result = response.json()
            analysis = result.get("response", "").strip()
            if analysis:
                return analysis
            else:
                return "Vision model processed the image but provided no description."
        else:
            return f"Vision model temporarily unavailable (Error: {response.status_code})"
                
    except requests.exceptions.Timeout:
        return "Vision analysis timed out. Please try again."
        
    except requests.exceptions.ConnectionError:
        return "Vision model is not available. Please ensure Ollama is running."
        
    except Exception as e:
        logger.error(f"Ollama query failed: {e}")
        return f"Vision analysis failed: {str(e)[:50]}..."


@router.post("/static-chat-with-image")
async def static_chat_with_image(
    question: str = Form(...),
    image: UploadFile = File(None)
):
    """Chat endpoint that supports both text and image inputs"""
    try:
        # If no image, use regular text chat
        if not image:
            return await static_chat({"question": question})
        
        # Validate image
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        # Read and process image
        file_content = await image.read()
        if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image file too large. Maximum size is 10MB.")
        
        try:
            pil_image = Image.open(io.BytesIO(file_content))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        
        # Extract text from image
        ocr_text = extract_text_from_image_simple(pil_image)
        logger.info(f"OCR extracted: {ocr_text[:100]}...")
        
        # Convert to base64 for vision model
        base64_image = image_to_base64_simple(pil_image)
        
        # Query vision model
        vision_response = await query_ollama_for_chat(base64_image, question, ocr_text)
        
        # Combine OCR and vision response
        if ocr_text and ocr_text != "No text detected in the image.":
            combined_response = f"I can see text in your image: '{ocr_text}'\n\n{vision_response}"
        else:
            combined_response = vision_response
        
        return {"answer": combined_response}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")


@router.post("/chat-multimodal")
async def chat_multimodal(
    question: str = Form(None),
    image: UploadFile = File(None)
):
    """
    Multimodal chat endpoint for handling text questions, images, or both
    """
    try:
        if not question and not image:
            raise HTTPException(status_code=400, detail="Please provide either a question or an image.")
        
        # Handle text-only question
        if question and not image:
            return await static_chat({"question": question})
        
        # Handle image processing
        if image:
            # Validate image
            if not image.content_type or not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
            
            # Read image
            file_content = await image.read()
            if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="Image file too large. Maximum size is 10MB.")
            
            try:
                pil_image = Image.open(io.BytesIO(file_content))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image file.")
            
            # Use Gemini Vision API instead of local models
            try:
                logger.info("Starting Gemini image analysis...")
                
                # Extract text with OCR
                ocr_text = extract_text_from_image_simple(pil_image)
                logger.info(f"OCR completed: {len(ocr_text) if ocr_text else 0} characters extracted")
                
                # Use Gemini for image analysis
                gemini_response = process_image_with_gemini_and_ocr(pil_image, question, ocr_text)
                
                logger.info("Gemini analysis completed successfully")
                return {"answer": gemini_response}
                    
            except Exception as e:
                logger.error(f"Gemini image processing failed: {type(e).__name__}: {str(e)}")
                # Provide a friendly fallback response
                return {"answer": f"I can see your image ({pil_image.size[0]}x{pil_image.size[1]} pixels) and it looks interesting! However, I'm having some technical difficulties analyzing it right now. Could you try asking a specific question about what you'd like to know, or try uploading the image again?"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multimodal chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")
