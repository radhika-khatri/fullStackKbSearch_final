#!/usr/bin/env python3
"""Test script to verify image processing with LLaVA model"""

import requests
import base64
import io
from PIL import Image, ImageDraw, ImageFont

def create_test_image():
    """Create a simple test image with text"""
    # Create a 400x200 image with white background
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    text = "Hello AppGallop!\nThis is a test image."
    draw.text((50, 50), text, fill='black', font=font)
    
    # Draw a simple rectangle
    draw.rectangle([50, 120, 150, 170], outline='blue', width=2)
    
    return img

def test_multimodal_endpoint():
    """Test the multimodal endpoint with a real image"""
    print("🧪 Testing multimodal endpoint with image...")
    
    # Create test image
    test_image = create_test_image()
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    try:
        # Send request to multimodal endpoint
        url = "http://localhost:8004/api/qa/chat-multimodal"
        
        files = {
            'image': ('test.png', img_buffer.getvalue(), 'image/png')
        }
        data = {
            'question': 'What do you see in this image?'
        }
        
        response = requests.post(url, files=files, data=data, timeout=30)
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!")
            print(f"Response: {result.get('answer', 'No answer found')}")
            return True
        else:
            print(f"❌ FAILED with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED with exception: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing image processing with LLaVA model...")
    success = test_multimodal_endpoint()
    if success:
        print("\n🎉 Image processing test completed successfully!")
    else:
        print("\n💥 Image processing test failed!")
