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

# Loyiha papkasidan real-estate.json faylini o'qish
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE_DIR, "real-estate.json")

try:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        REAL_ESTATE_SCHEMA = json.load(f)
except Exception as e:
    print(f"Xatolik: {SCHEMA_PATH} topilmadi! Iltimos real-estate.json faylini loyiha papkasiga joylashtiring.")
    REAL_ESTATE_SCHEMA = {}

# Endi mijoz faqat text yuboradi
class ExtractionRequest(BaseModel):
    text: str

@app.post("/extract")
async def extract_data(request: ExtractionRequest):
    if not REAL_ESTATE_SCHEMA:
        raise HTTPException(status_code=500, detail="Schema topilmadi, server xatoligi.")

    try:
        # Prompt modelga nima qilish kerakligini tushuntiradi
        prompt = f"Extract information from the following real estate listing text based on the provided JSON schema. Ensure you accurately identify the property category (e.g. Apartments, Houses) and map specific details to 'specs'.\n\nText:\n{request.text}"
        
        # Ollama orqali modelga so'rov yuborish
        response = ollama.chat(
            model='qwen-extractor',
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            format=REAL_ESTATE_SCHEMA,
            options={
                "temperature": 0.0,
            }
        )
        
        content = response['message']['content']
        
        # Qaytarilgan javobni JSON sifatida parse qilish
        try:
            extracted_json = json.loads(content)
            return {"status": "success", "data": extracted_json}
        except json.JSONDecodeError:
            cleaned_content = content.strip().replace('```json', '').replace('```', '')
            try:
                extracted_json = json.loads(cleaned_content)
                return {"status": "success", "data": extracted_json}
            except json.JSONDecodeError:
                return {
                    "status": "error", 
                    "message": "JSON parse qilishda xatolik yuz berdi. Model noto'g'ri format qaytardi.",
                    "raw_output": content
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
