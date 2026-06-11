from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import json
import os

app = FastAPI(
    title="Real Estate Data Extraction API",
    description="Qwen2.5-7b orqali real-estate.json schema asosida ko'chmas mulk ma'lumotlarini ajratuvchi API",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE_DIR, "real-estate.json")

REAL_ESTATE_SCHEMA = {}
CATEGORY_SCHEMAS = {}
CATEGORIES = []

try:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        REAL_ESTATE_SCHEMA = json.load(f)
        
    # Build category specific schemas
    base_item_properties = REAL_ESTATE_SCHEMA["items"]["properties"]["items"]["items"]["properties"]
    base_item_required = REAL_ESTATE_SCHEMA["items"]["properties"]["items"]["items"].get("required", [])
    
    CATEGORIES = REAL_ESTATE_SCHEMA["items"]["properties"]["name"]["enum"]
    
    all_of_conditions = REAL_ESTATE_SCHEMA["items"].get("allOf", [])
    
    for condition in all_of_conditions:
        cat_name = condition["if"]["properties"]["name"]["const"]
        cat_specs = condition["then"]["properties"]["items"]["items"]["properties"]["specs"]
        
        props = base_item_properties.copy()
        props["specs"] = cat_specs
        
        # Build specific schema for this category
        cat_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [cat_name]
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": props,
                            "required": base_item_required
                        }
                    }
                },
                "required": ["name", "items"]
            }
        }
        CATEGORY_SCHEMAS[cat_name] = cat_schema

except Exception as e:
    print(f"Xatolik: {SCHEMA_PATH} o'qishda muammo yuz berdi. {e}")

class ExtractionRequest(BaseModel):
    text: str

@app.post("/extract")
async def extract_data(request: ExtractionRequest):
    if not REAL_ESTATE_SCHEMA:
        raise HTTPException(status_code=500, detail="Schema topilmadi, server xatoligi.")

    try:
        # Step 1: Classification
        classification_prompt = f"Categorize the following real estate text into EXACTLY ONE of these categories: {', '.join(CATEGORIES)}. Respond ONLY with the category name, nothing else.\n\nText:\n{request.text}"
        
        class_response = ollama.chat(
            model='qwen-extractor',
            messages=[{'role': 'user', 'content': classification_prompt}],
            options={
                "temperature": 0.0,
                "num_predict": 10,  # We only need one word/phrase
            }
        )
        detected_category_raw = class_response['message']['content'].strip()
        
        detected_category = None
        for cat in CATEGORIES:
            if cat.lower() in detected_category_raw.lower():
                detected_category = cat
                break
                
        # Step 2: Extraction
        if detected_category and detected_category in CATEGORY_SCHEMAS:
            schema_to_use = CATEGORY_SCHEMAS[detected_category]
            extract_prompt = f"Extract information from the following real estate listing text based on the provided JSON schema. The property category is {detected_category}. Map specific details to 'specs'.\n\nText:\n{request.text}"
        else:
            # Fallback to the full schema if classification fails or is ambiguous
            schema_to_use = REAL_ESTATE_SCHEMA
            extract_prompt = f"Extract information from the following real estate listing text based on the provided JSON schema. Ensure you accurately identify the property category and map specific details to 'specs'.\n\nText:\n{request.text}"
        
        response = ollama.chat(
            model='qwen-extractor',
            messages=[{'role': 'user', 'content': extract_prompt}],
            format=schema_to_use,
            options={
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": 1024,
            }
        )
        
        content = response['message']['content']
        
        # Parsing JSON
        try:
            extracted_json = json.loads(content)
            return {"status": "success", "category_detected": detected_category, "data": extracted_json}
        except json.JSONDecodeError:
            cleaned_content = content.strip().replace('```json', '').replace('```', '')
            try:
                extracted_json = json.loads(cleaned_content)
                return {"status": "success", "category_detected": detected_category, "data": extracted_json}
            except json.JSONDecodeError:
                return {
                    "status": "error", 
                    "category_detected": detected_category,
                    "message": "JSON parse qilishda xatolik yuz berdi. Model noto'g'ri format qaytardi.",
                    "raw_output": content
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
