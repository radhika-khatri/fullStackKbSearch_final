#!/usr/bin/env python3
"""
Debug version of multimodal endpoint
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/debug-multimodal")
async def debug_multimodal(
    question: str = Form(None),
    image: UploadFile = File(None)
):
    """Debug multimodal endpoint with step-by-step logging"""
    print("🔍 DEBUG: Endpoint called")
    
    try:
        if not question and not image:
            print("❌ DEBUG: No question or image provided")
            raise HTTPException(status_code=400, detail="Please provide either a question or an image.")
        
        # Handle text-only question
        if question and not image:
            print(f"✅ DEBUG: Text-only question: {question}")
            return {"answer": f"Debug response to: {question}"}
        
        # Handle image processing
        if image:
            print(f"🖼️ DEBUG: Image received - {image.filename}, {image.content_type}")
            
            # Validate image
            if not image.content_type or not image.content_type.startswith('image/'):
                print("❌ DEBUG: Invalid file type")
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
            
            print("📖 DEBUG: Reading image bytes...")
            image_bytes = await image.read()
            print(f"📏 DEBUG: Image size: {len(image_bytes)} bytes")
            
            print("🖼️ DEBUG: Converting to PIL Image...")
            pil_image = Image.open(io.BytesIO(image_bytes))
            print(f"📐 DEBUG: PIL Image size: {pil_image.size}, mode: {pil_image.mode}")
            
            print("✅ DEBUG: Image processing completed successfully")
            
            return {
                "answer": f"Debug: Successfully processed image {image.filename} ({pil_image.size[0]}x{pil_image.size[1]})",
                "image_info": {
                    "filename": image.filename,
                    "size": pil_image.size,
                    "mode": pil_image.mode,
                    "bytes": len(image_bytes)
                }
            }
            
    except Exception as e:
        print(f"❌ DEBUG: Exception occurred: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting debug server on port 8005...")
    uvicorn.run(app, host="0.0.0.0", port=8005)
