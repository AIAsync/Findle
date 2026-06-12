"""
Regex-based data extractor for real estate listings.
Supports Uzbek (Latin & Cyrillic) and Russian languages.
"""

import re
from typing import Dict, Any, Optional


class RegexExtractor:
    """Extracts structured data from real estate text using regex patterns."""

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract all regex-matchable fields from text."""
        result = {}

        # ---- Numeric fields ----
        area = self._extract_total_area(text)
        if area is not None:
            result['total_area'] = area

        rooms = self._extract_rooms(text)
        if rooms is not None:
            result['rooms'] = rooms

        floor_info = self._extract_floor(text)
        if floor_info:
            result.update(floor_info)

        year = self._extract_year_built(text)
        if year is not None:
            result['year_built'] = year

        ceiling = self._extract_ceiling_height(text)
        if ceiling is not None:
            result['ceiling_height'] = ceiling

        land = self._extract_land_area(text)
        if land is not None:
            result['land_area'] = land

        building_area = self._extract_building_area(text)
        if building_area is not None:
            result['building_area'] = building_area

        # ---- Contact / Phone ----
        phone = self._extract_phone(text)
        if phone:
            result['contact'] = phone

        # ---- Price ----
        price = self._extract_price(text)
        if price is not None:
            result['price'] = price

        # ---- Boolean specs ----
        booleans = self._extract_booleans(text)
        if booleans:
            result['booleans'] = booleans

        # ---- Special fields ----
        office_class = self._extract_office_class(text)
        if office_class:
            result['office_class'] = office_class

        stars = self._extract_stars(text)
        if stars is not None:
            result['stars'] = stars

        total_rooms = self._extract_total_rooms(text)
        if total_rooms is not None:
            result['total_rooms'] = total_rooms

        parking_spaces = self._extract_parking_spaces(text)
        if parking_spaces is not None:
            result['parking_spaces'] = parking_spaces

        power = self._extract_power_capacity(text)
        if power is not None:
            result['power_capacity'] = power

        distance = self._extract_distance_to_city(text)
        if distance is not None:
            result['distance_to_city'] = distance

        neighbors = self._extract_neighbors(text)
        if neighbors is not None:
            result['neighbors'] = neighbors

        classrooms = self._extract_classrooms(text)
        if classrooms is not None:
            result['classrooms'] = classrooms

        total_beds = self._extract_total_beds(text)
        if total_beds is not None:
            result['total_beds'] = total_beds

        bathrooms = self._extract_bathrooms(text)
        if bathrooms is not None:
            result['bathrooms'] = bathrooms

        plumbing = self._extract_plumbing_points(text)
        if plumbing is not None:
            result['plumbing_points'] = plumbing

        total_spaces = self._extract_total_parking_spaces(text)
        if total_spaces is not None:
            result['total_spaces'] = total_spaces

        kitchen_area = self._extract_kitchen_area(text)
        if kitchen_area is not None:
            result['kitchen_area'] = kitchen_area

        hall_area = self._extract_hall_area(text)
        if hall_area is not None:
            result['hall_area'] = hall_area

        return result

    # ===============================
    # Area Extractors
    # ===============================

    def _extract_total_area(self, text: str) -> Optional[float]:
        """Extract total area in sq meters."""
        patterns = [
            # "85 kv.m", "85 кв.м", "85 m²", "85 м²"
            r'(\d+[\.,]?\d*)\s*(?:кв\.?\s*м|kv\.?\s*m|m²|м²|квадратн)',
            # "umumiy maydon 85", "общая площадь 85"
            r'(?:umumiy\s+)?(?:maydon[i]?|площад[ьи])\s*[:\-–]?\s*(\d+[\.,]?\d*)',
            # "85 kvadrat"
            r'(\d+[\.,]?\d*)\s*(?:kvadrat|квадрат)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = m.group(1).replace(',', '.')
                return float(val)
        return None

    def _extract_land_area(self, text: str) -> Optional[float]:
        """Extract land area in sotix."""
        patterns = [
            (r'(\d+[\.,]?\d*)\s*(?:sotix|сот(?:ок|ка|ки|их)?)', 'sotix'),
            (r'(\d+[\.,]?\d*)\s*(?:gektar|гектар|ga\b|га\b)', 'hectare'),
            (r'(?:yer\s*maydon[i]?|участ(?:ок|ка)|земельн)\s*[:\-–]?\s*(\d+[\.,]?\d*)\s*(?:кв\.?\s*м|kv\.?\s*m|m²|м²)', 'sqm'),
        ]
        for p, unit in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = float(m.group(1).replace(',', '.'))
                if unit == 'hectare':
                    return val * 100
                elif unit == 'sqm':
                    return val / 100
                return val
        return None

    def _extract_building_area(self, text: str) -> Optional[float]:
        """Extract building area."""
        patterns = [
            r'(?:bino\s*maydon[i]?|площадь\s*(?:дома|здания|строения)|building\s*area)\s*[:\-–]?\s*(\d+[\.,]?\d*)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return float(m.group(1).replace(',', '.'))
        return None

    def _extract_kitchen_area(self, text: str) -> Optional[float]:
        """Extract kitchen area."""
        patterns = [
            r'(?:oshxona|кухн[яи])\s*(?:maydon[i]?|площад[ьи])?\s*[:\-–]?\s*(\d+[\.,]?\d*)\s*(?:кв\.?\s*м|kv\.?\s*m|m²|м²)?',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return float(m.group(1).replace(',', '.'))
        return None

    def _extract_hall_area(self, text: str) -> Optional[float]:
        """Extract hall/dining area."""
        patterns = [
            r'(?:zal|зал)\s*(?:maydon[i]?|площад[ьи])?\s*[:\-–]?\s*(\d+[\.,]?\d*)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return float(m.group(1).replace(',', '.'))
        return None

    # ===============================
    # Rooms / Floor Extractors
    # ===============================

    def _extract_rooms(self, text: str) -> Optional[int]:
        """Extract number of rooms."""
        patterns = [
            r'(\d+)\s*[-–]?\s*(?:xona(?:li)?|хона(?:ли)?)',
            r'(\d+)\s*[-–]?\s*(?:комнат(?:ная|ный|ных|а|ы)?|комн\.?)',
            r'(?:xonalar\s+soni|количество\s+комнат)\s*[:\-–]?\s*(\d+)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _extract_floor(self, text: str) -> Dict[str, int]:
        """Extract floor and total floors."""
        result = {}

        # "4/9", "4/9 qavat", "4/9 этаж"
        patterns_combined = [
            r'(\d+)\s*/\s*(\d+)\s*[-–]?\s*(?:qavat|этаж|etaj|qavatda|этаже)?',
            r'(\d+)\s+(?:из|dan)\s+(\d+)\s*(?:qavat|этаж)?',
        ]
        for p in patterns_combined:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                floor = int(m.group(1))
                total = int(m.group(2))
                if floor <= total <= 200:
                    result['floor'] = floor
                    result['total_floors'] = total
                    return result

        # Single floor: "4-qavat", "на 4 этаже"
        patterns_floor = [
            r'(\d+)\s*[-–]?\s*(?:qavat(?:da)?|этаж[ае]?|й\s*этаж)',
            r'(?:на|da)\s+(\d+)\s*[-–]?\s*(?:qavat|этаж)',
        ]
        for p in patterns_floor:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if val <= 200:
                    result['floor'] = val
                    break

        # Total floors: "9 qavatli", "9-этажный"
        patterns_total = [
            r'(\d+)\s*[-–]?\s*(?:qavatli|этажн(?:ый|ая|ое|ого)|qavatlik)',
        ]
        for p in patterns_total:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if val <= 200:
                    result['total_floors'] = val
                    break

        return result

    # ===============================
    # Contact / Price
    # ===============================

    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number."""
        patterns = [
            r'(\+?998\s*[-–]?\s*\d{2}\s*[-–]?\s*\d{3}\s*[-–]?\s*\d{2}\s*[-–]?\s*\d{2})',
            r'(\+?7\s*[-–]?\s*\d{3}\s*[-–]?\s*\d{3}\s*[-–]?\s*\d{2}\s*[-–]?\s*\d{2})',
            r'(8\s*[-–]?\s*\d{3}\s*[-–]?\s*\d{3}\s*[-–]?\s*\d{2}\s*[-–]?\s*\d{2})',
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                phone = re.sub(r'[\s\-–]', '', m.group(1))
                return phone
        return None

    def _extract_price(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract price and currency."""
        patterns = [
            (r'(?:\$\s*)([\d\s.,]+)', 'USD'),
            (r'([\d\s.,]+)\s*(?:\$|dollar|доллар|у\.?\s*е\.?)', 'USD'),
            (r"([\d\s.,]+)\s*(?:so['\u2018\u2019]?m|сум|сўм|sum)", 'UZS'),
            (r'(?:narx[i]?|цена|стоимость|price)\s*[:\-–]?\s*([\d\s.,]+)', None),
        ]
        for p, currency in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val_str = m.group(1).strip().replace(' ', '').replace(',', '.')
                try:
                    val = float(val_str)
                    result = {'amount': val}
                    if currency:
                        result['currency'] = currency
                    return result
                except ValueError:
                    continue
        return None

    # ===============================
    # Time / Height Extractors
    # ===============================

    def _extract_year_built(self, text: str) -> Optional[int]:
        """Extract year built."""
        patterns = [
            r'(?:qurilgan|построен|постройки)\s*(?:yili|в|года?)?\s*[:\-–]?\s*((?:19|20)\d{2})',
            r'((?:19|20)\d{2})\s*[-–]?\s*(?:yil(?:da|i)?|год(?:а|у)?|г\.?)\s*(?:qurilgan|построен|постройки)?',
            r'(?:qurilish|строительство)\s*(?:yili|год)\s*[:\-–]?\s*((?:19|20)\d{2})',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                year = int(m.group(1))
                if 1800 <= year <= 2030:
                    return year
        return None

    def _extract_ceiling_height(self, text: str) -> Optional[float]:
        """Extract ceiling height in meters."""
        patterns = [
            r'(?:shift\s*balandligi|потолк[иов]|высота\s*потолков?|ceiling)\s*[:\-–]?\s*(\d+[\.,]?\d*)\s*(?:m|м|metr)?',
            r'(\d+[\.,]?\d*)\s*(?:m|м)\s*(?:shift|потолок|потолки)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = float(m.group(1).replace(',', '.'))
                if 1.5 <= val <= 10.0:
                    return val
        return None

    # ===============================
    # Boolean Extractors
    # ===============================

    def _extract_booleans(self, text: str) -> Dict[str, bool]:
        """Extract boolean indicators for UZ/RU."""
        result = {}
        text_lower = text.lower()

        boolean_patterns = {
            'furnished': {
                'positive': [r'mebel(?:li|langan)', r'мебел(?:ирован|ь\s+есть)', r'с\s+мебелью', r'mebel\s+bor'],
                'negative': [r'mebel(?:siz)', r'без\s+мебели', r"mebel\s+yo['\u2018\u2019]?q", r'мебели\s+нет'],
            },
            'gas': {
                'positive': [r'gaz(?:\s+bor|lashtirilgan|li)', r'газ(?:\s+есть|ифициров)', r'газ\s+подведен'],
                'negative': [r"gaz\s+yo['\u2018\u2019]?q", r'газ(?:а)?\s+нет', r'без\s+газа'],
            },
            'water': {
                'positive': [r'suv\s+bor', r'вода\s+есть', r'водоснабжение', r"suv\s+ta['\u2018\u2019]?minot"],
                'negative': [r"suv\s+yo['\u2018\u2019]?q", r'воды?\s+нет', r'без\s+воды'],
            },
            'electricity': {
                'positive': [r'elektr(?:\s+bor|lashtirilgan|ik)', r'электричеств(?:о\s+есть|ом)', r'свет\s+есть', r'svet\s+bor', r'tok\s+bor'],
                'negative': [r"elektr\s+yo['\u2018\u2019]?q", r'электричества?\s+нет', r"svet\s+yo['\u2018\u2019]?q"],
            },
            'heating': {
                'positive': [r'isitish(?:\s+tizimi)?\s+bor', r'отоплени(?:е\s+есть|ем)', r'с\s+отоплением'],
                'negative': [r"isitish\s+yo['\u2018\u2019]?q", r'отопления?\s+нет', r'без\s+отопления'],
            },
            'internet': {
                'positive': [r'internet\s+bor', r'интернет\s+есть', r'с\s+интернетом', r'wi[-\s]?fi'],
                'negative': [r"internet\s+yo['\u2018\u2019]?q", r'интернета?\s+нет'],
            },
            'security': {
                'positive': [r"qo['\u2018\u2019]?riqlanadigan", r'охран(?:яемый|а\s+есть|яемая)', r'с\s+охраной', r'xavfsizlik'],
                'negative': [r'охраны?\s+нет', r'без\s+охраны'],
            },
            'parking': {
                'positive': [r'parkovka\s+bor', r'парковка\s+есть', r'с\s+парковкой', r'avto\s*turargoh'],
                'negative': [r"parkovka\s+yo['\u2018\u2019]?q", r'парковки?\s+нет'],
            },
            'sewage': {
                'positive': [r'kanalizatsiya\s+bor', r'канализаци(?:я\s+есть|ей)', r'с\s+канализацией'],
                'negative': [r"kanalizatsiya\s+yo['\u2018\u2019]?q", r'канализации?\s+нет'],
            },
            'road_access': {
                'positive': [r"yo['\u2018\u2019]?l\s+bor", r'подъезд(?:ная\s+дорога|ной)', r'дорога\s+есть', r"asfalt\s+yo['\u2018\u2019]?l"],
                'negative': [r"yo['\u2018\u2019]?l\s+yo['\u2018\u2019]?q", r'дороги?\s+нет'],
            },
            'ramp_access': {
                'positive': [r'ramp(?:a)?\s+bor', r'рамп(?:а\s+есть|ой)', r'с\s+рампой'],
                'negative': [],
            },
            'railway_access': {
                'positive': [r"temir\s*yo['\u2018\u2019]?l", r'ж[/.]?\s*д\s*(?:путь|подъезд|ветка)', r'железнодорожн'],
                'negative': [],
            },
            'pool': {
                'positive': [r'basseyn', r'бассейн', r'hovuz', r'хавуз'],
                'negative': [],
            },
            'gym': {
                'positive': [r'sport\s*zal', r'спорт\s*зал', r'тренажерн'],
                'negative': [],
            },
            'pit': {
                'positive': [r'яма\s+есть', r'с\s+ямой', r'chuqur\s+bor', r'смотровая\s+яма'],
                'negative': [],
            },
            'exhaust_system': {
                'positive': [r'вытяжк', r'exhaust', r'shamollatish\s+tizimi'],
                'negative': [],
            },
            'ventilation': {
                'positive': [r'вентиляци', r'shamollatish', r'ventilya[tc]i'],
                'negative': [],
            },
            'conference_hall': {
                'positive': [r'конференц\s*зал', r'konferents\s*zal', r'conference\s*hall'],
                'negative': [],
            },
            'kitchen': {
                'positive': [r'oshxona\s+bor', r'кухня\s+есть', r'с\s+кухней'],
                'negative': [r"oshxona\s+yo['\u2018\u2019]?q", r'кухни?\s+нет', r'без\s+кухни'],
            },
            'fence': {
                'positive': [r'devori?\s+bor', r'забор', r'огорож', r'с\s+забором'],
                'negative': [],
            },
            'irrigation': {
                'positive': [r"sug['\u2018\u2019]?orish", r'ирригаци', r'поливн', r'suvori'],
                'negative': [],
            },
            'frontage': {
                'positive': [r'фасад', r'fasad', r"ko['\u2018\u2019]?cha\s+tomon", r'красная\s+линия'],
                'negative': [],
            },
            'window_display': {
                'positive': [r'витрин', r'vitrina'],
                'negative': [],
            },
            'truck_parking': {
                'positive': [r'yuk\s*mashinalar?\s*uchun', r'для\s+(?:грузовиков|фур)', r'truck\s*parking'],
                'negative': [],
            },
            'locker_rooms': {
                'positive': [r'раздевалк', r'kiyinish\s*xona'],
                'negative': [],
            },
        }

        for field, patterns in boolean_patterns.items():
            found = False
            for neg_pattern in patterns.get('negative', []):
                if re.search(neg_pattern, text_lower):
                    result[field] = False
                    found = True
                    break
            if not found:
                for pos_pattern in patterns.get('positive', []):
                    if re.search(pos_pattern, text_lower):
                        result[field] = True
                        break

        return result

    # ===============================
    # Special Fields
    # ===============================

    def _extract_office_class(self, text: str) -> Optional[str]:
        """Extract office class."""
        patterns = [
            r'(?:klass|класс|class)\s*[:\-–]?\s*(A\+|A|B\+|B|C|D)\b',
            r'\b(A\+|B\+)\s*[-–]?\s*(?:klass|класс|class)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        return None

    def _extract_stars(self, text: str) -> Optional[int]:
        """Extract hotel star rating."""
        patterns = [
            r'(\d)\s*[-–]?\s*(?:yulduz(?:li)?|звезд|star)',
            r'(?:yulduz|звезд|star)\s*[:\-–]?\s*(\d)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 1 <= val <= 5:
                    return val
        return None

    def _extract_total_rooms(self, text: str) -> Optional[int]:
        """Extract total rooms (for hotels)."""
        patterns = [
            r'(?:jami|всего|total)\s+(\d+)\s*(?:xona|номер|room)',
            r'(\d+)\s*(?:номер(?:ов|а)?|nomer)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if val >= 5:
                    return val
        return None

    def _extract_parking_spaces(self, text: str) -> Optional[int]:
        """Extract parking spaces count."""
        patterns = [
            r'(\d+)\s*(?:ta\s+)?(?:parkovka\s+joy|парковочн|parking\s*space)',
            r'(?:parkovka|парковка|parking)\s*[:\-–]?\s*(\d+)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _extract_power_capacity(self, text: str) -> Optional[float]:
        """Extract power capacity in kW."""
        patterns = [
            r'(\d+[\.,]?\d*)\s*(?:кВт|kVt|kvt|kW)',
            r'(?:quvvat|мощность|power)\s*[:\-–]?\s*(\d+[\.,]?\d*)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return float(m.group(1).replace(',', '.'))
        return None

    def _extract_distance_to_city(self, text: str) -> Optional[float]:
        """Extract distance to city in km."""
        patterns = [
            r'(?:shahardan|от\s+города|shahargacha)\s*(\d+[\.,]?\d*)\s*(?:km|км)',
            r'(\d+[\.,]?\d*)\s*(?:km|км)\s*(?:shahardan|от\s+города|shahargacha)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return float(m.group(1).replace(',', '.'))
        return None

    def _extract_neighbors(self, text: str) -> Optional[int]:
        """Extract number of neighbors (for rooms)."""
        patterns = [
            r'(\d+)\s*(?:ta\s+)?(?:qo[\'\u2018\u2019]?shni|сосед)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _extract_classrooms(self, text: str) -> Optional[int]:
        """Extract number of classrooms."""
        patterns = [
            r'(\d+)\s*(?:ta\s+)?(?:sinf\s*xona|класс(?:ная\s*комната)?|classroom)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _extract_total_beds(self, text: str) -> Optional[int]:
        """Extract total beds (for hostels)."""
        patterns = [
            r'(\d+)\s*(?:ta\s+)?(?:joy|кроват|кровать|bed|o[\'\u2018\u2019]?rin)',
            r'(?:jami|всего)\s+(\d+)\s*(?:joy|кроват|bed)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _extract_bathrooms(self, text: str) -> Optional[int]:
        """Extract number of bathrooms."""
        patterns = [
            r'(\d+)\s*(?:ta\s+)?(?:hammom|ванн(?:ая|ых)|bathroom|sanitar)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _extract_plumbing_points(self, text: str) -> Optional[int]:
        """Extract plumbing points (for salons)."""
        patterns = [
            r'(\d+)\s*(?:ta\s+)?(?:suv\s*nuqta|водоточк|plumbing)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _extract_total_parking_spaces(self, text: str) -> Optional[int]:
        """Extract total parking spaces count (for parking lots)."""
        patterns = [
            r'(?:jami|всего)\s+(\d+)\s*(?:ta\s+)?(?:joy|мест|space)',
            r'(\d+)\s*(?:ta\s+)?(?:avto\s*joy|машиномест)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None
