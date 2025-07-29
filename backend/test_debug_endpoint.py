#!/usr/bin/env python3
"""
Debug endpoint to test image processing step by step
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
import io
import uvicorn

app = FastAPI()

@app.post("/debug-image")
async def debug_image_processing(
    question: str = Form(None),
    image: UploadFile = File(None)
):
    """Debug image processing step by step"""
    try:
        print("🚀 Debug endpoint called")
        
        if not image:
            return {"status": "no_image", "message": "No image provided"}
        
        print(f"📁 Image received: {image.filename}, type: {image.content_type}")
        
        # Read image data
        image_data = await image.read()
        print(f"📏 Image size: {len(image_data)} bytes")
        
        # Try to open with PIL
        try:
            pil_image = Image.open(io.BytesIO(image_data))
            print(f"🖼️ PIL image opened: {pil_image.size}, mode: {pil_image.mode}")
        except Exception as e:
            print(f"❌ PIL error: {e}")
            return {"status": "pil_error", "error": str(e)}
        
        # Try basic OCR (without complex config)
        try:
            import pytesseract
            print("📝 Starting basic OCR...")
            ocr_text = pytesseract.image_to_string(pil_image)
            print(f"✅ OCR completed: '{ocr_text[:50]}...'")
        except Exception as e:
            print(f"❌ OCR error: {e}")
            return {"status": "ocr_error", "error": str(e)}
        
        return {
            "status": "success",
            "image_size": len(image_data),
            "pil_size": pil_image.size,
            "ocr_text": ocr_text[:100] + "..." if len(ocr_text) > 100 else ocr_text,
            "question": question
        }
        
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🔧 Starting debug server on port 8005...")
    uvicorn.run(app, host="0.0.0.0", port=8005)
