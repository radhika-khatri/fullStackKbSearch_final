#!/usr/bin/env python3
"""
Simple image test with curl equivalent
"""
import subprocess
import tempfile
from PIL import Image, ImageDraw

def test_simple_image():
    """Test with a real small image file"""
    print("🚀 Creating and testing simple image...")
    
    # Create a small test image
    img = Image.new('RGB', (100, 50), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 15), "Hello World", fill='black')
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img.save(tmp.name, 'JPEG')
        tmp_path = tmp.name
    
    print(f"📁 Saved image to: {tmp_path}")
    
    # Test with curl
    cmd = [
        'curl', '-X', 'POST',
        'http://localhost:8004/api/qa/chat-multimodal',
        '-F', f'image=@{tmp_path}',
        '-F', 'question=What do you see?',
        '--max-time', '10'
    ]
    
    print("📤 Testing with curl...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        print(f"📊 Exit code: {result.returncode}")
        if result.stdout:
            print(f"✅ Response: {result.stdout}")
        if result.stderr:
            print(f"⚠️ Error: {result.stderr}")
        
        # Cleanup
        import os
        os.unlink(tmp_path)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_simple_image()
