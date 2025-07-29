#!/usr/bin/env python3
"""
Test image processing step by step to isolate the issue
"""
import requests
import io
import time
from PIL import Image, ImageDraw

def test_step_by_step():
    """Test each step to find the bottleneck"""
    print("🚀 Testing image processing step by step...")
    
    # Create a very small test image
    img = Image.new('RGB', (50, 25), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), "Hi", fill='black')
    
    # Save to minimal bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=50)
    image_bytes = buffer.getvalue()
    
    print(f"📏 Image size: {len(image_bytes)} bytes")
    
    # Test 1: Just image upload without question (should skip LLaVA)
    print("\n🧪 Test 1: Image upload only (no question)")
    files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
    
    start_time = time.time()
    try:
        response = requests.post(
            "http://localhost:8004/api/qa/chat-multimodal",
            files=files,
            timeout=20
        )
        elapsed = time.time() - start_time
        print(f"⏱️ Time taken: {elapsed:.2f} seconds")
        print(f"📊 Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response: {result.get('answer', 'No answer')[:100]}...")
        else:
            print(f"❌ Error: {response.text[:200]}...")
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"⏰ Timed out after {elapsed:.2f} seconds")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.2f} seconds: {e}")
    
    print("\n" + "="*50)
    
    # Test 2: Image + simple question (will use LLaVA)
    print("\n🧪 Test 2: Image + simple question")
    files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
    data = {'question': 'Hi'}
    
    start_time = time.time()
    try:
        response = requests.post(
            "http://localhost:8004/api/qa/chat-multimodal",
            data=data,
            files=files,
            timeout=45  # Longer timeout for LLaVA
        )
        elapsed = time.time() - start_time
        print(f"⏱️ Time taken: {elapsed:.2f} seconds")
        print(f"📊 Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response: {result.get('answer', 'No answer')[:100]}...")
        else:
            print(f"❌ Error: {response.text[:200]}...")
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"⏰ Timed out after {elapsed:.2f} seconds")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.2f} seconds: {e}")

if __name__ == "__main__":
    test_step_by_step()
