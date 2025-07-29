"""
Add minimal test endpoint to existing main.py
"""
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

# This should be added to your existing main.py
@app.post("/test-upload")
async def test_upload(
    image: UploadFile = File(None)
):
    """Minimal test endpoint for file upload"""
    try:
        if not image:
            return {"status": "no_file", "message": "No file provided"}
        
        # Just read the file bytes
        file_content = await image.read()
        
        return {
            "status": "success",
            "filename": image.filename,
            "content_type": image.content_type,
            "size": len(file_content)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
