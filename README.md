# Qwen2.5 Data Extractor API

Bu loyiha Ollama orqali ishlaydigan, Qwen2.5-7b modelining siqilgan (q4_K_M) versiyasidan foydalanib, berilgan matndan JSON schema asosida ma'lumotlarni ajratib oluvchi API hisoblanadi.
Model tez ishlashga optimallashtirilgan, "thinking" qismi o'chirilgan va faqat toza JSON qaytarishga majburlangan.

## VPS da ishga tushirish bo'yicha qo'llanma

### 1. Ollama'ni o'rnatish
Agar VPS da Ollama o'rnatilmagan bo'lsa, uni o'rnating:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Python va Pip'ni tekshirish
Python 3.8+ va pip o'rnatilganligiga ishonch hosil qiling.

### 3. Loyihani ishga tushirish
Loyihaning asosiy papkasiga kiring (`extractor_api`) va `start.sh` faylini ishga tushiring:

```bash
chmod +x start.sh
./start.sh
```

Bu skript avtomatik tarzda:
1. Kerakli python kutubxonalarini o'rnatadi.
2. `qwen2.5:7b-instruct-q4_K_M` siqilgan modelini yuklab oladi.
3. `Modelfile` orqali tez ishlash va faqat JSON qaytarish uchun moslashtirilgan `qwen-extractor` modelini yaratadi.
4. FastAPI serverini `8000`-portda ishga tushiradi.

### 4. API dan foydalanish (Test qilish)

API ishlagach, unga POST so'rovini yuborishingiz mumkin. Quyida curl orqali misol keltirilgan:

```bash
curl -X 'POST' \
  'http://localhost:8000/extract' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Kecha iPhone 15 Pro Max 256GB qora ranglisini 1200 dollarga sotib oldim. Juda zo'\''r telefon ekan.",
  "json_schema": {
    "type": "object",
    "properties": {
      "category": {
        "type": "string",
        "description": "Mahsulot kategoriyasi (masalan: Elektronika, Kiyim-kechak va hk.)"
      },
      "brand": {
        "type": "string"
      },
      "model": {
        "type": "string"
      },
      "storage": {
        "type": "string"
      },
      "color": {
        "type": "string"
      },
      "price_usd": {
        "type": "number"
      }
    },
    "required": ["category", "brand", "model", "price_usd"]
  }
}'
```

**Kutilayotgan javob:**
```json
{
  "status": "success",
  "data": {
    "category": "Elektronika",
    "brand": "Apple",
    "model": "iPhone 15 Pro Max",
    "storage": "256GB",
    "color": "qora",
    "price_usd": 1200
  }
}
```
