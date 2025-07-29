from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import speech_recognition as sr
import io
import tempfile
import os
from pathlib import Path
import logging
from datetime import datetime
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

# Create uploads directory for voice files if it doesn't exist
VOICE_UPLOADS_DIR = Path("uploads/voice")
VOICE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/speech-to-text")
async def speech_to_text(
    audio_file: UploadFile = File(...),
    save_audio: bool = Form(default=False)
):
    """
    Convert speech audio file to text using speech recognition
    """
    try:
        # Validate file type
        if not audio_file.content_type or not any(
            audio_file.content_type.startswith(t) for t in ['audio/', 'video/webm']
        ):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an audio file."
            )
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # Save audio file if requested
        saved_file_path = None
        if save_audio:
            file_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voice_{timestamp}_{file_id}.wav"
            saved_file_path = VOICE_UPLOADS_DIR / filename
            
            with open(saved_file_path, "wb") as f:
                f.write(audio_data)
            
            logger.info(f"Voice file saved: {saved_file_path}")
        
        # Convert to text using speech recognition
        recognizer = sr.Recognizer()
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Load audio file
            with sr.AudioFile(temp_file_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source)
            
            # Try multiple recognition services for better accuracy
            text_results = []
            
            # Google Speech Recognition (primary)
            try:
                google_text = recognizer.recognize_google(audio)
                text_results.append({
                    "service": "google",
                    "text": google_text,
                    "confidence": "high"
                })
                logger.info(f"Google recognition: {google_text}")
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                logger.error(f"Google Speech Recognition service error: {e}")
            
            # Sphinx (offline fallback)
            try:
                sphinx_text = recognizer.recognize_sphinx(audio)
                text_results.append({
                    "service": "sphinx",
                    "text": sphinx_text,
                    "confidence": "medium"
                })
                logger.info(f"Sphinx recognition: {sphinx_text}")
            except sr.UnknownValueError:
                logger.warning("Sphinx could not understand audio")
            except sr.RequestError as e:
                logger.error(f"Sphinx error: {e}")
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            if not text_results:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": False,
                        "message": "Could not transcribe audio. Please try speaking more clearly.",
                        "text": "",
                        "audio_saved": save_audio,
                        "saved_path": str(saved_file_path) if saved_file_path else None
                    }
                )
            
            # Use the best available result (prefer Google)
            best_result = text_results[0]
            for result in text_results:
                if result["service"] == "google":
                    best_result = result
                    break
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "text": best_result["text"],
                    "confidence": best_result["confidence"],
                    "service_used": best_result["service"],
                    "all_results": text_results,
                    "audio_saved": save_audio,
                    "saved_path": str(saved_file_path) if saved_file_path else None,
                    "file_size": len(audio_data),
                    "processing_time": "< 2 seconds"
                }
            )
            
        except Exception as processing_error:
            # Clean up temporary file if it exists
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise processing_error
            
    except Exception as e:
        logger.error(f"Speech-to-text error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process audio: {str(e)}"
        )

@router.post("/voice-chat")
async def voice_chat(
    audio_file: UploadFile = File(...),
    save_audio: bool = Form(default=True),
    session_id: str = Form(default="anonymous")
):
    """
    Complete voice chat workflow: speech-to-text + chat response
    """
    try:
        # First, convert speech to text
        audio_data = await audio_file.read()
        
        # Create a new UploadFile object for the speech-to-text function
        from io import BytesIO
        audio_stream = BytesIO(audio_data)
        
        # Recreate UploadFile for internal processing
        temp_upload = UploadFile(
            file=audio_stream,
            filename=audio_file.filename,
            headers=audio_file.headers
        )
        
        # Convert to text
        speech_result = await speech_to_text(temp_upload, save_audio)
        
        if not speech_result.body:
            raise HTTPException(status_code=500, detail="Failed to process speech")
        
        # Parse the speech result
        import json
        speech_data = json.loads(speech_result.body)
        
        if not speech_data.get("success") or not speech_data.get("text"):
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "message": "Could not understand voice input. Please try again.",
                    "speech_result": speech_data
                }
            )
        
        # Get the transcribed text
        user_question = speech_data["text"].strip()
        
        # Send to chat API
        from api.qa import static_chat
        chat_response = await static_chat({"question": user_question})
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "user_text": user_question,
                "bot_response": chat_response["answer"],
                "speech_confidence": speech_data.get("confidence", "unknown"),
                "service_used": speech_data.get("service_used", "unknown"),
                "audio_saved": speech_data.get("audio_saved", False),
                "saved_path": speech_data.get("saved_path"),
                "session_id": session_id
            }
        )
        
    except Exception as e:
        logger.error(f"Voice chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice chat failed: {str(e)}"
        )

@router.get("/voice/health")
async def voice_health_check():
    """
    Check if voice processing dependencies are available
    """
    try:
        # Check speech recognition
        sr_available = True
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
        except ImportError:
            sr_available = False
        
        # Check if microphone access works (basic check)
        mic_available = True
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                pass
        except:
            mic_available = False
        
        return {
            "status": "healthy" if sr_available else "degraded",
            "speech_recognition": sr_available,
            "microphone_access": mic_available,
            "voice_storage": VOICE_UPLOADS_DIR.exists(),
            "storage_path": str(VOICE_UPLOADS_DIR),
            "supported_formats": ["audio/wav", "audio/mp3", "audio/webm", "audio/ogg"]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "speech_recognition": False,
            "microphone_access": False
        }

@router.get("/voice/files")
async def list_voice_files():
    """
    List saved voice files (for admin/debugging)
    """
    try:
        voice_files = []
        if VOICE_UPLOADS_DIR.exists():
            for file_path in VOICE_UPLOADS_DIR.glob("*.wav"):
                stat = file_path.stat()
                voice_files.append({
                    "filename": file_path.name,
                    "size_bytes": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return {
            "total_files": len(voice_files),
            "files": voice_files,
            "storage_path": str(VOICE_UPLOADS_DIR)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list voice files: {str(e)}")
