import pandas as pd
import json
from pathlib import Path
import PyPDF2
import base64
from PIL import Image
from io import BytesIO
import math
import numpy as np

# ==========================================
# 💡 Support Assistant Project .gitignore
# For FastAPI + PostgreSQL + React + Docker
# Author: Shivam Srivastav
# ==========================================
# File: backend/api/files.py    
# ==========================================

def clean_json(obj):
        if isinstance(obj, dict):
            return {k: clean_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_json(item) for item in obj]
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        elif isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
            return None
        else:
            return obj

def parse_file(file_path: Path, limit: int = 10):
    """
    Parses files (.csv, .xlsx, .json, .txt, .pdf, image) and returns a preview of first `limit` records or metadata.
    author: 
        shivam srivastav
        date: 2025-07-01
    Args:
        file_path (Path): Path to the uploaded file
        limit (int): Max number of records or preview lines

    Returns:
        dict: {
            "filename": str,
            "type": str,
            "preview": list or dict (depends on type)
        }
    """
    ext = file_path.suffix.lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
            preview_data = df.head(limit).to_dict(orient="records")

        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
            preview_data = df.head(limit).to_dict(orient="records")

        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    preview_data = data[:limit]
                elif isinstance(data, dict):
                    preview_data = [{k: data[k]} for k in list(data.keys())[:limit]]
                else:
                    raise ValueError("Unsupported JSON structure")

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                preview_data = [{"line": line.strip()} for line in lines[:limit]]

        elif ext == ".pdf":
            preview_data = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in range(min(limit, len(reader.pages))):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    preview_data.append({
                        "page": page_num + 1,
                        "text": text.strip() if text else "[Empty]"
                    })

        elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
            with Image.open(file_path) as img:
                buffered = BytesIO()
                img.save(buffered, format=img.format)
                encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                preview_data = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "preview_base64": f"data:image/{img.format.lower()};base64,{encoded_image}"
                }

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return {
            "filename": file_path.name,
            "type": ext[1:],  # e.g. jpg
            "preview": clean_json(preview_data)
        }

    except Exception as e:
        raise RuntimeError(f"❌ Failed to parse file: {str(e)}")


def extract_full_text(file_path: Path):
    """
    Extract full text content from files for embedding generation.
    
    Args:
        file_path (Path): Path to the uploaded file
        
    Returns:
        str: Full text content of the file
    """
    ext = file_path.suffix.lower()
    
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
            # Convert dataframe to text representation
            return df.to_string()
            
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
            return df.to_string()
            
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # For knowledge base JSON files, extract questions and answers
                    text_parts = []
                    for item in data:
                        if isinstance(item, dict):
                            if 'question' in item and 'answer' in item:
                                text_parts.append(f"Q: {item['question']}\nA: {item['answer']}")
                            else:
                                text_parts.append(str(item))
                    return "\n\n".join(text_parts)
                else:
                    return json.dumps(data, indent=2)
                    
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
                
        elif ext == ".pdf":
            full_text = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
            return "\n".join(full_text)
            
        else:
            raise ValueError(f"Unsupported file type for text extraction: {ext}")
            
    except Exception as e:
        raise RuntimeError(f"❌ Failed to extract full text: {str(e)}")
