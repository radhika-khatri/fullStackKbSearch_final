#!/usr/bin/env python3
"""
Create a test image and test the multimodal endpoint
"""
import requests
import io
from PIL import Image, ImageDraw, ImageFont
import time

def create_test_image():
    """Create a simple test image with text"""
    # Create a white background image
    img = Image.new('RGB', (300, 150), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    text = "Hello AppGallop!\nThis is a test image."
    draw.text((20, 40), text, fill='black')
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()

def test_multimodal_with_image():
    """Test the multimodal endpoint with an actual image"""
    print("🖼️ Creating test image...")
    image_bytes = create_test_image()
    print(f"📏 Image size: {len(image_bytes)} bytes")
    
    # Prepare the request
    files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
    data = {'question': 'What do you see in this image?'}
    
    print("📤 Sending image to multimodal endpoint...")
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8004/api/qa/chat-multimodal",
            data=data,
            files=files,
            timeout=30  # 30 second timeout
        )
        
        elapsed = time.time() - start_time
        print(f"⏱️ Response time: {elapsed:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"🤖 AI Response: {result.get('answer', 'No answer')}")
            return True
        else:
            print(f"❌ Error {response.status_code}:")
            print(f"   {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"⏰ Request timed out after {elapsed:.2f} seconds")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.2f} seconds: {e}")
        return False

def test_image_only():
    """Test with image only (no question)"""
    print("\n" + "="*50)
    print("🧪 Testing image-only upload...")
    
    image_bytes = create_test_image()
    files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
    
    print("📤 Sending image without question...")
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8004/api/qa/chat-multimodal",
            files=files,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        print(f"⏱️ Response time: {elapsed:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"🤖 AI Response: {result.get('answer', 'No answer')}")
            return True
        else:
            print(f"❌ Error {response.status_code}:")
            print(f"   {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"⏰ Request timed out after {elapsed:.2f} seconds")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.2f} seconds: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing multimodal image processing...\n")
    
    # Test 1: Image with question
    success1 = test_multimodal_with_image()
    
    # Test 2: Image only
    success2 = test_image_only()
    
    print("\n" + "="*50)
    print("📊 Test Results:")
    print(f"Image + Question: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Image Only: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 or success2:
        print("\n🎉 Image processing is working!")
    else:
        print("\n⚠️ Image processing needs more work...")
