#!/bin/bash

# Venv orqali ishga tushirish
if [ ! -d "venv" ]; then
    echo "0. Virtual environment yaratilmoqda..."
    python3 -m venv venv
fi

echo "1. Kerakli paketlarni o'rnatish..."
source venv/bin/activate
pip install torch --no-cache-dir --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt

echo "2. API serverni ishga tushirish..."
echo "   Modellar birinchi ishga tushirishda yuklab olinadi (GLiNER + E5-small)"
uvicorn main:app --host 0.0.0.0 --port 8000
