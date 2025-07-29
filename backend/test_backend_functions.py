#!/usr/bin/env python3
"""
Test our exact backend function with a real image
"""
import requests
import io
import base64
from PIL import Image, ImageDraw, ImageFont

def image_to_base64_simple(image_file):
    """Exact copy of the backend function"""
    try:
        if hasattr(image_file, 'read'):
            img_data = image_file.read()
            image_file.seek(0)
        else:
            img_data = image_file
        
        if not img_data:
            raise ValueError("Empty image data received")
        
        # Convert to PIL Image for processing
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Compress image for faster processing
        width, height = img.size
        if width > 800 or height > 800:
            ratio = min(800/width, 800/height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to JPEG with compression
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=60, optimize=True)
        buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        if not img_base64:
            raise ValueError("Base64 conversion resulted in empty string")
            
        return img_base64
        
    except Exception as e:
        raise Exception(f"Failed to process image: {str(e)}")

async def query_ollama_for_chat(base64_image: str, question: str, ocr_text: str) -> str:
    """Exact copy of the backend function"""
    try:
        # Simple, focused prompt optimized for LLaVA
        if question and question.strip():
            prompt = question.strip()[:150]  # Shorter prompt for LLaVA
        else:
            prompt = "What do you see?"  # Very simple default
        
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
        return "Vision analysis timed out - image may be too complex"
    except Exception as e:
        return f"Vision analysis failed: {str(e)}"

def test_with_backend_functions():
    """Test with our exact backend functions"""
    # Create test image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 80), "Hello LLaVA!", fill='black')
    
    # Convert to bytes like a file upload
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    print("🔄 Testing with backend functions...")
    
    try:
        # Step 1: Convert to base64 with our function
        base64_img = image_to_base64_simple(buffer)
        print(f"✅ Base64 conversion successful: {len(base64_img)} chars")
        
        # Step 2: Query LLaVA with our function
        print("📤 Querying LLaVA...")
        import asyncio
        result = asyncio.run(query_ollama_for_chat(
            base64_img, 
            "Describe what you see in this image", 
            ""
        ))
        
        print(f"✅ LLaVA result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_with_backend_functions()
