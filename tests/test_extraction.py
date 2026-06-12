import sys
import os
import pytest

# Add parent directory to path so we can import extractors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractors.pipeline import ExtractionPipeline

@pytest.fixture(scope="module")
def pipeline():
    return ExtractionPipeline()

def test_uzbek_apartment(pipeline):
    text = "Toshkent shahar, Chilonzor tumani, 3 xonali kvartira, 85 kv.m, 4/9 qavatda, evro ta'mir, mebelli, g'isht bino, narxi 75000$. Tel: +998901234567"
    
    result = pipeline.extract(text)
    
    assert result['name'] == "Apartments"
    item = result['items'][0]
    
    # regex fields
    assert item['total_area'] == 85.0
    assert item['specs']['rooms'] == 3
    assert item['specs']['floor'] == 4
    assert item['specs']['total_floors'] == 9
    assert item['price']['amount'] == 75000.0
    assert item['price']['currency'] == "USD"
    assert item['contact'] == "+998901234567"
    
    # booleans
    assert item['specs']['furnished'] == True
    
    # enums
    assert item['specs']['condition'] == "Euro repair"
    assert item['specs']['building_type'] == "Brick"
    
    # NER
    assert "Toshkent" in item['region']
    assert "Chilonzor" in item['district']

def test_russian_apartment(pipeline):
    text = "Продается 2-комнатная квартира в Ташкенте, Юнусабадский район, 65 кв.м, 3/12 этаж, панельный дом, евроремонт, с мебелью. Цена: 55000$. Тел: +998901234567"
    
    result = pipeline.extract(text)
    
    assert result['name'] == "Apartments"
    item = result['items'][0]
    
    assert item['total_area'] == 65.0
    assert item['specs']['rooms'] == 2
    assert item['specs']['floor'] == 3
    assert item['specs']['total_floors'] == 12
    assert item['price']['amount'] == 55000.0
    
    assert item['specs']['furnished'] == True
    assert item['specs']['condition'] == "Euro repair"
    assert item['specs']['building_type'] == "Panel"
    
    assert "Ташкент" in item['region']
    assert "Юнусабадский" in item['district']

def test_house(pipeline):
    text = "Hovli sotiladi, 6 sotix, Qibray tumani. 5 xona, evro remont, gaz, suv, svet bor. 120000$"
    
    result = pipeline.extract(text)
    
    assert result['name'] == "Houses"
    item = result['items'][0]
    
    assert item['total_area'] == 6.0
    assert item['specs']['rooms'] == 5
    assert item['price']['amount'] == 120000.0
    
    assert item['specs']['gas'] == True
    assert item['specs']['water'] == True
    assert item['specs']['electricity'] == True
    assert item['specs']['condition'] == "Euro repair"
    assert item['transaction_type'] == "Sale"
