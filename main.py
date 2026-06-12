import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from extractors.pipeline import ExtractionPipeline

app = FastAPI(
    title="Real Estate Data Extraction API",
    description="Regex, NER va Embedding orqali real-estate.json schema asosida ko'chmas mulk ma'lumotlarini ajratuvchi API",
    version="2.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE_DIR, "real-estate.json")

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("Modellar yuklanmoqda...")
    pipeline = ExtractionPipeline()
    print("API ishga tayyor!")

class ExtractionRequest(BaseModel):
    text: str

@app.post("/extract")
async def extract_data(request: ExtractionRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline hali tayyor emas.")

    try:
        # Extract data using the pipeline
        extracted_data = pipeline.extract(request.text)
        
        return {
            "status": "success", 
            "category_detected": extracted_data.get("name"), 
            "data": extracted_data
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
