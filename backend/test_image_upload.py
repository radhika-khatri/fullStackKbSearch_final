#!/usr/bin/env python3
"""
Test script to validate image upload and processing functionality
"""
import requests
import io
import base64
from PIL import Image, ImageDraw, ImageFont

def create_test_image():
    """Create a simple test image with text"""
    # Create a small image with text
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a system font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Add some text
    draw.text((20, 50), "Hello AppGallop!", fill='black', font=font)
    draw.text((20, 100), "This is a test image", fill='blue', font=font)
    draw.text((20, 150), "for OCR processing", fill='red', font=font)
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    return img_buffer.getvalue()

def test_image_upload():
    """Test image upload to the multimodal chat endpoint"""
    print("🖼️  Creating test image...")
    image_data = create_test_image()
    
    print("📤 Testing image upload...")
    url = "http://localhost:8004/api/qa/chat-multimodal"
    
    # Prepare the multipart form data
    files = {
        'image': ('test_image.png', image_data, 'image/png')
    }
    data = {
        'question': 'What text do you see in this image?'
    }
    
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Response: {result['answer']}")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (this is expected for first-time model loading)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health_check():
    """Test the image processing health endpoint"""
    print("🏥 Testing health check...")
    url = "http://localhost:8004/api/image/health"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Health check status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Health status: {result}")
            
            if result.get('status') == 'healthy':
                print("✅ All services healthy!")
                return True
            else:
                print("⚠️  Some services may have issues")
                return False
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting image processing tests...\n")
    
    # Test health first
    health_ok = test_health_check()
    print()
    
    if health_ok:
        # Test actual image upload
        upload_ok = test_image_upload()
        print()
        
        if upload_ok:
            print("🎉 All tests passed! Image processing is working correctly.")
        else:
            print("⚠️  Image upload test failed, but health check passed.")
            print("This might be due to model loading time or other issues.")
    else:
        print("❌ Health check failed. Please check if:")
        print("   - Backend is running on port 8004")
        print("   - Ollama is running and has LLaVA model")
        print("   - Tesseract is installed")
