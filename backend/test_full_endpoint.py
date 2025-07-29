#!/usr/bin/env python3
"""
Test the full multimodal endpoint with debugging
"""
import requests
import json
from PIL import Image, ImageDraw
import io
import base64

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (300, 150), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.text((50, 60), "Test Image for API", fill='black')
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

def test_multimodal_endpoint():
    """Test the full multimodal endpoint with detailed logging"""
    print("🧪 Testing full multimodal endpoint...\n")
    
    # Create test image
    image_file = create_test_image()
    
    # Prepare the request
    files = {
        'file': ('test.png', image_file, 'image/png')
    }
    
    data = {
        'question': 'What do you see in this image?',
        'chat_session_id': 'test_session_123'
    }
    
    try:
        print("📤 Sending request to /api/qa/chat-multimodal...")
        print(f"   Question: {data['question']}")
        print(f"   Session ID: {data['chat_session_id']}")
        
        # Send request with extended timeout
        response = requests.post(
            'http://localhost:8000/api/qa/chat-multimodal',
            files=files,
            data=data,
            timeout=60  # Extended timeout for debugging
        )
        
        print(f"📥 Response received!")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: ~{response.elapsed.total_seconds():.1f}s")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"   OCR Text: {result.get('ocr_text', 'None')[:100]}...")
            print(f"   Vision Analysis: {result.get('vision_analysis', 'None')[:100]}...")
            print(f"   Final Answer: {result.get('answer', 'None')[:100]}...")
        else:
            print(f"❌ Error Response:")
            print(f"   Content: {response.text[:500]}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out after 60 seconds")
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - is the server running?")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_multimodal_endpoint()
