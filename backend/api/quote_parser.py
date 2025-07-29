
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.gemini_vision import parse_quote_table_with_gemini
import logging
import traceback

router = APIRouter()


class OCRTextRequest(BaseModel):
    ocr_text: str


@router.post("/parse-quote-table")
def parse_quote_table(request: OCRTextRequest):
    logger = logging.getLogger("api.quote_parser")
    logger.info(f"Received OCR text for table parsing (length={len(request.ocr_text)}): {request.ocr_text[:200]}...")
    try:
        logger.info("Calling Gemini table extraction...")
        result = parse_quote_table_with_gemini(request.ocr_text)
        logger.info(f"Gemini table extraction result: {result}")
        return {"table": result}
    except Exception as e:
        logger.error(f"Exception in /api/parse-quote-table: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Gemini table extraction failed: {e}")
