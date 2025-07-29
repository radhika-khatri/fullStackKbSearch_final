import re
from typing import Dict, Optional

def is_photo_image(image: 'PIL.Image.Image') -> bool:
    """Heuristic: returns True if image is likely a photo (not a digital image)."""
    # Check for EXIF data (photos usually have it)
    exif = getattr(image, '_getexif', lambda: None)()
    if exif:
        return True
    # Heuristic: photos have more noise, less sharp edges
    # (Could use ML, but keep it simple for now)
    return False

def extract_quote_fields(ocr_text: str) -> Optional[Dict]:
    """Parse OCR text for quote/invoice fields. Returns dict if found, else None."""
    # Look for keywords
    if not ocr_text:
        return None
    text = ocr_text.lower()
    if not ("quote" in text or "invoice" in text):
        return None
    # Extract fields (very basic, can be improved)
    fields = {}
    # Date
    date_match = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
    if date_match:
        fields["date"] = date_match.group(1)
    # Total/Amount
    total_match = re.search(r"total\s*[:=]?\s*([\d,.]+)", text)
    if total_match:
        fields["total"] = total_match.group(1)
    # Items (look for lines with qty, price, etc.)
    items = []
    for line in ocr_text.splitlines():
        if re.search(r"\d+\s+\w+\s+[\d,.]+", line):
            items.append(line.strip())
    if items:
        fields["items"] = items
    # Vendor/Customer (first line or lines with name)
    lines = ocr_text.splitlines()
    if lines:
        fields["vendor"] = lines[0]
    return fields if fields else None
