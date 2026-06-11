from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import json

app = FastAPI(
    title="Data Extraction API",
    description="Qwen2.5-7b orqali ma'lumotlarni berilgan JSON schema bo'yicha ajratib oluvchi API",
    version="1.0.0"
)

class ExtractionRequest(BaseModel):
    text: str
    json_schema: dict

@app.post("/extract")
async def extract_data(request: ExtractionRequest):
    try:
        # Prompt modelga nima qilish kerakligini tushuntiradi
        prompt = f"Extract information from the following text based on the provided JSON schema.\n\nText:\n{request.text}"
        
        # Ollama orqali modelga so'rov yuborish
        # Ollama'ning 'format' parametriga JSON schema beriladi, shunda model faqat shu schemaga mos JSON qaytarishga majbur bo'ladi
        response = ollama.chat(
            model='qwen-extractor',
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            format=request.json_schema,
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
            # Ba'zida model baribir markdown qoldirgan bo'lsa tozalashga urinib ko'ramiz
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
