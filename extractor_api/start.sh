#!/bin/bash

echo "1. Kerakli paketlarni o'rnatish..."
pip install -r requirements.txt

echo "2. Ollama uchun Qwen2.5 (siqilgan q4_K_M) modelini yuklash..."
ollama pull qwen2.5:7b-instruct-q4_K_M

echo "3. Custom model (qwen-extractor) yaratish..."
ollama create qwen-extractor -f Modelfile

echo "4. API serverni ishga tushirish..."
uvicorn main:app --host 0.0.0.0 --port 8000
