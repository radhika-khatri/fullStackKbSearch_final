

# By Radhika 

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import re
import cv2
import ollama

app = FastAPI()

GST_PATTERN = re.compile(
    r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][A-Z0-9]Z[A-Z0-9]$", re.IGNORECASE
)

def is_valid_gst(gst: str) -> bool:
    valid = bool(GST_PATTERN.fullmatch(gst.strip().upper()))
    print(f"✅ GST Validation: {gst} → {valid}")
    return valid

def preprocess_image(image_path, processed_path="preprocessed.jpg"):
    print(f"🖼️ Reading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image.")
        return None
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    border_size = 32
    image = cv2.copyMakeBorder(
        image, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    cv2.imwrite(processed_path, image)
    print(f"✅ Image preprocessed and saved to: {processed_path}")
    return processed_path

def extract_raw_text_from_image(image_path):
    processed_path = preprocess_image(image_path)
    if processed_path is None:
        return None
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    print("🧠 Sending image to LLM for OCR...")
    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': (
                "You are an OCR engine. Extract all visible text from this image. "
                "Return only the raw text exactly as it appears in the image."
            ),
            'images': [image_bytes]
        }]
    )
    raw_text = response['message']['content']
    print(f"📄 Raw OCR Text:\n{raw_text}\n")
    return raw_text

def extract_gst_number(text):
    print("🔍 Searching for GST number...")
    match = re.search(r"(GST(?:IN)?\s*[:\-]?\s*)([0-9A-Z]{15})", text, re.IGNORECASE)
    if match:
        gst = match.group(2).strip()
        print(f"✅ Extracted GST Number: {gst}")
        return gst
    print("❌ GST number not found.")
    return None

def extract_relevant_info_with_mistral(text):
    prompt = (
        "You will be given text extracted from a shipping document. "
        "Extract the following fields:\n"
        "- Shipped From\n- Shipped To\n- Quantity\n- Contact Number(s)\n"
        "Return it as a plain list like:\n"
        "Shipped From: ...\nShipped To: ...\nQuantity: ...\nContact Numbers: ..."
    )
    print("🧠 Sending extracted text to Mistral for structured info extraction...")
    response = ollama.chat(
        model='mistral',
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': text}
        ]
    )
    result = response['message']['content']
    print(f"📦 Mistral Extracted Info:\n{result}\n")
    return result

def parse_mistral_response(response_text):
    details = {}
    print("📑 Parsing Mistral response...")
    for line in response_text.splitlines():
        if ':' in line:
            key, val = line.split(':', 1)
            details[key.strip()] = val.strip()
    print(f"✅ Parsed Details: {details}")
    return details

@app.post("/extract-invoice")
async def extract_invoice(
    file: UploadFile = File(...),
    manual_gst: str = Form(None),
    shipped_from: str = Form(None),
    shipped_to: str = Form(None),
    quantity: str = Form(None),
    contact_numbers: str = Form(None)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        file_path = temp_file.name
        temp_file.write(await file.read())
        print(f"📂 Saved uploaded file to: {file_path}")

    try:
        raw_text = extract_raw_text_from_image(file_path)
        if not raw_text:
            raise HTTPException(status_code=500, detail="Failed to extract text from image.")

        extracted_gst = extract_gst_number(raw_text)
        mistral_text = extract_relevant_info_with_mistral(raw_text)
        extracted_details = parse_mistral_response(mistral_text)

        if extracted_gst and is_valid_gst(extracted_gst):
            gst_final = extracted_gst
        elif manual_gst and is_valid_gst(manual_gst):
            gst_final = manual_gst.strip().upper()
        else:
            raise HTTPException(status_code=422, detail="Valid GST Number not found. Please provide manually.")

        final_details = {
            "Shipped From": shipped_from or extracted_details.get("Shipped From", ""),
            "Shipped To": shipped_to or extracted_details.get("Shipped To", ""),
            "Quantity": quantity or extracted_details.get("Quantity", ""),
            "Contact Numbers": contact_numbers or extracted_details.get("Contact Numbers", ""),
            "GST Number": gst_final
        }

        print(f"✅ Final Output Data: {final_details}")
        return JSONResponse(content={"status": "success", "data": final_details})

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🧹 Deleted temp file: {file_path}")