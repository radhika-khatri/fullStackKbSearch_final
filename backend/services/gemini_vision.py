# Example function to call Gemini and parse quote/invoice table
import os
from typing import List, Dict

def parse_quote_table_with_gemini(ocr_text: str) -> List[Dict]:
    import logging
    logger = logging.getLogger(__name__)
    """
    Use Gemini to extract a structured table from OCR text of a quote/invoice.
    Returns a list of dicts with keys: description, qty, unit, unit_price, discount, net_price, etc.
    """
    # You should replace this with your actual Gemini API call
    # Here is a pseudo-implementation:
    import google.generativeai as genai
    import logging
    import json
    logger = logging.getLogger(__name__)
    # Use env variable if set, else fallback to hardcoded key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyB5V3qgB25MFkv79JGaHUH75G047iQ5VIU"
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"""
    Extract ONLY the line items table from the following OCR text of a quote or invoice. Return ONLY a JSON array of objects with these fields: description, qty, unit, unit_price, discount, net_price. If a field is missing, use null. Only include actual line items, not totals or notes. DO NOT return any explanation, summary, or extra text. Output ONLY the JSON array, wrapped in a markdown code block like this:\n```json\n[{{...}}, ...]\n```.\n\nOCR TEXT:\n{ocr_text}
    """

    import re, json
    response = model.generate_content(prompt)
    text = response.text.strip()
    logger.info(f"Gemini raw response: {text}")
    # Try to extract JSON array from code block
    match = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    table = None
    if match:
        table_json = match.group(1)
        try:
            table = json.loads(table_json)
        except Exception:
            pass
    if table is None:
        # fallback: try to find any JSON array in the response
        match = re.search(r"(\[.*?\])", text, re.DOTALL)
        if match:
            try:
                table = json.loads(match.group(1))
            except Exception:
                pass
    # If table is still None, try to build table using vector similarity from OCR text
    if table is None or not isinstance(table, list) or len(table) == 0:
        from sentence_transformers import SentenceTransformer, util
        import numpy as np
        FIELDS = ["description", "qty", "unit", "unit_price", "discount", "net_price"]
        FIELD_SYNONYMS = {
            "description": ["description", "item", "product", "details", "name"],
            "qty": ["qty", "quantity", "amount", "number", "count"],
            "unit": ["unit", "units", "pcs", "piece", "type"],
            "unit_price": ["unit price", "price", "rate", "cost per unit", "per item"],
            "discount": ["discount", "rebate", "offer"],
            "net_price": ["net price", "total", "amount", "final", "subtotal"]
        }
        model = SentenceTransformer('all-MiniLM-L6-v2')
        lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
        logger.info(f"OCR lines: {lines}")
        # Try to find table-like lines (with multiple columns)
        candidate_rows = []
        for line in lines:
            # Split by |, tab, or 2+ spaces
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
            elif '\t' in line:
                parts = [p.strip() for p in line.split('\t')]
            else:
                parts = [p.strip() for p in re.split(r'\s{2,}', line)]
            if len(parts) >= 2 and len(parts) <= len(FIELDS):
                candidate_rows.append(parts)
        logger.info(f"Candidate table rows: {candidate_rows}")
        # If no candidate rows, just return one row with all text as description
        if not candidate_rows:
            logger.info("No candidate rows found, returning all text as description.")
            return [{"description": ocr_text, "qty": None, "unit": None, "unit_price": None, "discount": None, "net_price": None}]
        # For each row, assign parts to fields using vector similarity
        field_phrases = [syn for field in FIELDS for syn in FIELD_SYNONYMS[field]]
        field_embeds = model.encode(field_phrases)
        normalized_table = []
        for row_parts in candidate_rows:
            row_dict = {field: None for field in FIELDS}
            part_embeds = model.encode(row_parts)
            used_fields = set()
            for i, part in enumerate(row_parts):
                sims = util.cos_sim(part_embeds[i], field_embeds)[0].cpu().numpy()
                best_idx = int(np.argmax(sims))
                # Map best_idx to field
                field_idx = best_idx // max(1, len(FIELD_SYNONYMS[FIELDS[best_idx // len(FIELD_SYNONYMS)]]))
                best_field = FIELDS[min(field_idx, len(FIELDS)-1)]
                if best_field in used_fields:
                    # Find next best unused field
                    sorted_idx = np.argsort(sims)[::-1]
                    for alt_idx in sorted_idx:
                        alt_field_idx = alt_idx // max(1, len(FIELD_SYNONYMS[FIELDS[alt_idx // len(FIELD_SYNONYMS)]]))
                        alt_field = FIELDS[min(alt_field_idx, len(FIELDS)-1)]
                        if alt_field not in used_fields:
                            best_field = alt_field
                            break
                row_dict[best_field] = part
                used_fields.add(best_field)
            logger.info(f"Row assignment: {row_dict}")
            normalized_table.append(row_dict)
        logger.info(f"Final normalized table: {normalized_table}")
        return normalized_table
    # Enforce fixed schema for each row (if table was found)
    FIELDS = ["description", "qty", "unit", "unit_price", "discount", "net_price"]
    normalized_table = []
    for row in table:
        normalized_row = {field: row.get(field, None) for field in FIELDS}
        normalized_table.append(normalized_row)
    return normalized_table
"""
Gemini Vision API integration for image analysis
"""
import google.generativeai as genai
import base64
import io
from PIL import Image
import logging

# Configure Gemini
GEMINI_API_KEY = "AIzaSyB5V3qgB25MFkv79JGaHUH75G047iQ5VIU"
genai.configure(api_key=GEMINI_API_KEY)

logger = logging.getLogger(__name__)

def analyze_image_with_gemini(pil_image, question=None):
    """
    Analyze image using Google Gemini Vision API with user-friendly responses
    """
    try:
        logger.info("Starting Gemini image analysis...")
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare the prompt for natural, conversational responses
        if question and question.strip():
            prompt = f"""
You are a helpful AI assistant analyzing an image for a user. They asked: "{question}"

Please provide a friendly, natural response that:
1. Directly answers their question in a conversational way
2. Describes what you see in the image clearly and simply
3. Mentions any text or important details you notice
4. Keeps the tone warm and helpful

Write as if you're having a conversation with the user, not giving a technical report. Be specific but friendly.
"""
        else:
            prompt = """
You are a helpful AI assistant analyzing an image for a user. 

Please describe what you see in a friendly, conversational way. Imagine you're helping a friend understand the image:

1. Start with a warm greeting and overview of what the image shows
2. Describe the main elements in simple, clear language
3. Mention any text, colors, or interesting details you notice
4. End with an offer to help if they have specific questions

Keep it natural and engaging, like you're having a conversation!
"""
        
        # Generate response
        response = model.generate_content([prompt, pil_image])
        
        if response and response.text:
            logger.info("Gemini analysis completed successfully")
            return response.text.strip()
        else:
            logger.warning("Gemini returned empty response")
            return "I can see your image, but I'm having trouble analyzing it right now. Could you try asking a specific question about what you'd like to know?"
            
    except Exception as e:
        logger.error(f"Gemini analysis failed: {type(e).__name__}: {str(e)}")
        return f"I can see you've shared an image, but I'm having some technical difficulties analyzing it right now. Please try again in a moment!"

def process_image_with_gemini_and_ocr(pil_image, question=None, ocr_text=None):
    """
    Process image with both Gemini vision and OCR, combining results in a user-friendly format
    """
    try:
        logger.info("Processing image with Gemini + OCR...")
        
        # Get Gemini analysis
        gemini_analysis = analyze_image_with_gemini(pil_image, question)
        
        # Create a natural, friendly response format
        if ocr_text and ocr_text.strip() and ocr_text not in ["No text detected in the image.", "Failed to extract text from image."]:
            # When there's readable text, integrate it naturally
            if question and any(word in question.lower() for word in ['text', 'read', 'writing', 'written', 'says', 'word']):
                # User specifically asked about text
                combined_response = f"""{gemini_analysis}

📝 **Text I can read in the image:**
"{ocr_text.strip()}"

Hope this helps! Let me know if you'd like me to focus on any specific part of the image."""
            else:
                # General analysis with text included naturally
                combined_response = f"""{gemini_analysis}

I can also see there's some text in the image that reads: "{ocr_text.strip()}"

Feel free to ask if you'd like me to focus on any particular aspect!"""
        else:
            # No readable text found
            combined_response = f"""{gemini_analysis}

I don't see any readable text in this image, but I'm happy to answer any other questions about what I can see!"""
        
        logger.info("Gemini + OCR processing completed")
        return combined_response
        
    except Exception as e:
        logger.error(f"Combined processing failed: {e}")
        return f"I can see your image! While I had some technical issues with the detailed analysis, I'm here to help. Could you tell me what specifically you'd like to know about the image?"
