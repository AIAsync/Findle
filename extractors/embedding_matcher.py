"""
Embedding-based semantic matcher using multilingual-e5-small (via FastEmbed).
Uses cosine similarity to match input text to categories and enum values.
Supports Uzbek and Russian through multilingual embeddings + synonym lists.
"""

import numpy as np
from fastembed import TextEmbedding
from typing import Dict, Any, Optional, List, Tuple


# ===============================================================
# Category synonyms: UZ (Latin), UZ (Cyrillic), Russian, English
# ===============================================================
CATEGORY_SYNONYMS = {
    "Apartments": [
        "kvartira", "xonadon", "квартира", "apartment", "flat",
        "ko'p qavatli uy", "многоквартирный дом",
        "1 xonali kvartira", "2 xonali kvartira", "3 xonali kvartira",
        "однокомнатная квартира", "двухкомнатная квартира", "трёхкомнатная квартира",
    ],
    "Houses": [
        "uy", "hovli", "hovli uy", "xususiy uy", "частный дом", "дом",
        "house", "коттедж частный", "жилой дом", "hovli joy",
    ],
    "Townhouses": [
        "taunxaus", "таунхаус", "townhouse", "qator uy",
        "блокированный дом", "сблокированный дом",
    ],
    "Rooms": [
        "xona", "комната", "room", "yotoq xona", "жилая комната",
        "комната в общежитии",
    ],
    "Cottages": [
        "kottej", "dacha", "коттедж", "дача", "cottage",
        "загородный дом", "shahar tashqarisidagi uy",
    ],
    "Offices": [
        "ofis", "офис", "office", "ish joyi", "рабочий кабинет",
        "офисное помещение", "biznes markaz", "бизнес центр",
    ],
    "Retail space": [
        "savdo maydoni", "торговая площадь", "retail space",
        "торговое помещение", "savdo joyi", "do'kon uchun joy",
        "коммерческое помещение",
    ],
    "Restaurants and cafes": [
        "restoran", "kafe", "oshxona", "ресторан", "кафе",
        "restaurant", "cafe", "общепит", "ovqatlanish joyi",
    ],
    "Hotels": [
        "mehmonxona", "гостиница", "отель", "hotel",
        "гостиничный комплекс", "hotel kompleksi",
    ],
    "Factories": [
        "fabrika", "zavod", "фабрика", "завод", "factory",
        "производственное помещение", "ishlab chiqarish", "цех", "sex",
    ],
    "Warehouses": [
        "ombor", "склад", "warehouse", "omborxona",
        "складское помещение", "saqlash joyi",
    ],
    "Logistics centers": [
        "logistika markazi", "логистический центр", "logistics center",
        "logistika", "распределительный центр",
    ],
    "Agricultural land": [
        "qishloq xo'jaligi yeri", "ekin maydoni",
        "сельскохозяйственная земля", "сельхоз земля",
        "agricultural land", "fermer yeri", "пашня",
    ],
    "Residential land": [
        "turar-joy yeri", "uy-joy uchun yer", "yashash uchun yer",
        "жилая земля", "земля под ИЖС", "residential land",
        "uy qurish uchun yer",
    ],
    "Commercial land": [
        "tijorat yeri", "biznes uchun yer",
        "коммерческая земля", "земля коммерческого назначения",
        "commercial land",
    ],
    "Medical centers": [
        "tibbiyot markazi", "klinika", "shifoxona",
        "медицинский центр", "клиника", "больница",
        "medical center", "kasalxona",
    ],
    "Educational institutions": [
        "ta'lim muassasasi", "maktab", "universitet",
        "учебное заведение", "школа", "университет",
        "educational institution", "o'quv markazi", "учебный центр",
    ],
    "Sports complexes": [
        "sport kompleksi", "sport zal", "fitness",
        "спортивный комплекс", "спортзал", "фитнес",
        "sports complex", "stadion", "стадион",
    ],
    "Garages": [
        "garaj", "гараж", "garage", "avtomobil uchun joy",
        "гаражный бокс",
    ],
    "Parking spaces": [
        "parkovka joyi", "парковочное место", "parking space",
        "avto turargoh joyi", "машиноместо",
    ],
    "Parking lots": [
        "parkovka maydoni", "avtoturargoh", "автостоянка",
        "parking lot", "стоянка", "парковка большая",
    ],
    "Salons": [
        "salon", "go'zallik saloni", "салон", "салон красоты",
        "beauty salon", "sartaroshxona", "парикмахерская",
    ],
    "Shops and boutiques": [
        "do'kon", "butik", "magazin", "магазин", "бутик",
        "shop", "boutique", "sotish joyi", "торговая точка",
    ],
    "Plots": [
        "yer", "uchastka", "yer uchastka", "участок",
        "земельный участок", "plot", "tomorqa",
    ],
    "Hostels": [
        "yotoqxona", "xostel", "хостел", "общежитие",
        "hostel", "yashash joyi obshaga",
    ],
}


# ===============================================================
# Enum field synonyms for semantic matching
# ===============================================================
ENUM_SYNONYMS = {
    "property_type": {
        "Residential": [
            "turar-joy", "yashash", "жилой", "жилая", "residential",
            "uy-joy", "yashash uchun", "жилое помещение",
        ],
        "Commercial": [
            "tijorat", "коммерческий", "коммерческая", "commercial",
            "biznes", "savdo", "бизнес", "торговый",
        ],
        "Industrial": [
            "sanoat", "ishlab chiqarish", "промышленный", "industrial",
            "производственный", "завод", "фабрика",
        ],
        "Agricultural": [
            "qishloq xo'jaligi", "сельскохозяйственный", "agricultural",
            "dehqonchilik", "fermer", "фермерский",
        ],
        "Land": [
            "yer", "земля", "земельный", "land", "участок", "yer uchastka",
        ],
        "Other": [
            "boshqa", "другой", "other", "прочее",
        ],
    },
    "transaction_type": {
        "Sale": [
            "sotish", "sotuv", "sotiladi", "продажа", "продаётся",
            "продается", "sale", "sell", "sotaman",
        ],
        "Rent": [
            "ijara", "ijaraga", "аренда", "сдаётся", "сдается",
            "rent", "ijaraga beraman", "в аренду", "ijaraga beradi",
        ],
        "Lease": [
            "uzok muddatli ijara", "долгосрочная аренда", "lease",
            "лизинг", "lizing",
        ],
        "Exchange": [
            "almashtirish", "ayirboshlash", "обмен", "exchange",
            "almashaman", "меняю",
        ],
    },
    "payment_terms": {
        "Cash": [
            "naqd", "naqdiga", "наличные", "cash", "нал",
            "naqd pul", "наличными",
        ],
        "Mortgage": [
            "ipoteka", "ипотека", "mortgage", "kredit", "кредит",
        ],
        "Installment": [
            "bo'lib to'lash", "рассрочка", "installment",
            "to'lov muddati", "в рассрочку", "rassrochka",
        ],
        "Exchange": [
            "almashtirish", "ayirboshlash", "обмен", "exchange",
            "бартер", "barter",
        ],
        "Other": [
            "boshqa", "другой", "other",
        ],
    },
    "condition": {
        "Rough": [
            "qora suvoq", "черновая отделка", "rough",
            "qora", "черновая", "без отделки",
        ],
        "White box": [
            "oq suvoq", "предчистовая", "white box",
            "чистовая без ремонта",
        ],
        "Cosmetic repair": [
            "oddiy ta'mir", "косметический ремонт", "cosmetic repair",
            "простой ремонт", "oddiy remont",
        ],
        "Euro repair": [
            "evro ta'mir", "evro remont", "евроремонт",
            "euro repair", "еврота'мир", "evremont",
        ],
        "Designer repair": [
            "dizaynerlik ta'mir", "дизайнерский ремонт",
            "designer repair", "эксклюзивный ремонт", "dizayner remont",
        ],
        "Needs repair": [
            "ta'mir talab", "требует ремонта", "needs repair",
            "ta'mirga muhtoj", "нуждается в ремонте",
        ],
    },
    "building_type": {
        "Brick": [
            "g'isht", "кирпич", "кирпичный", "brick",
            "g'ishtli", "g'isht bino",
        ],
        "Panel": [
            "panel", "панель", "панельный", "panel bino",
        ],
        "Monolithic": [
            "monolit", "монолит", "монолитный", "monolithic",
        ],
        "Block": [
            "blok", "блок", "блочный", "block",
        ],
        "Wood": [
            "yog'och", "деревянный", "wood", "wooden", "деревянное",
        ],
        "Other": [
            "boshqa", "другой", "other",
        ],
    },
    "storage_type": {
        "Dry": ["quruq", "сухой", "dry", "quruq ombor"],
        "Cold": ["sovuq", "холодный", "cold", "sovuq ombor"],
        "Freezing": ["muzlatgich", "морозильный", "freezing"],
        "Hazardous": ["xavfli", "опасный", "hazardous", "xavfli moddalar"],
        "Heated": ["isitilgan", "отапливаемый", "heated", "issiq ombor"],
    },
    "zoning": {
        "Commercial": ["tijorat", "коммерческий", "commercial"],
        "Industrial": ["sanoat", "промышленный", "industrial"],
        "Mixed-use": ["aralash", "смешанное", "mixed-use", "mixed use"],
        "Other": ["boshqa", "другой", "other"],
    },
    "traffic": {
        "High": [
            "yuqori oqim", "высокая проходимость", "high traffic",
            "gavjum", "ko'p odamli", "оживлённый",
        ],
        "Medium": ["o'rta oqim", "средняя проходимость", "medium traffic"],
        "Low": [
            "past oqim", "низкая проходимость", "low traffic",
            "tinch", "тихий",
        ],
    },
    "garage_type": {
        "Attached": ["uyga biriktirilgan", "пристроенный", "attached"],
        "Detached": ["alohida", "отдельно стоящий", "detached"],
        "Underground": ["yer osti", "подземный", "underground"],
        "Metal": ["temir", "металлический", "metal"],
        "Brick": ["g'isht", "кирпичный", "brick"],
        "Other": ["boshqa", "другой", "other"],
    },
    "soil_quality": {
        "Excellent": ["a'lo", "отличное", "excellent"],
        "Good": ["yaxshi", "хорошее", "good"],
        "Average": ["o'rta", "среднее", "average"],
        "Poor": ["yomon", "past", "плохое", "poor"],
    },
    "bathroom_type": {
        "Private": ["shaxsiy", "отдельный", "private", "o'z hammomi"],
        "Shared": ["umumiy", "общий", "shared"],
        "None": ["yo'q", "нет", "none"],
    },
    "indoor_outdoor": {
        "Indoor": ["yopiq", "крытый", "закрытый", "indoor"],
        "Outdoor": ["ochiq", "открытый", "outdoor"],
        "Covered": ["qopqoqli", "навес", "covered", "yopilgan"],
    },
    "surface_type": {
        "Asphalt": ["asfalt", "асфальт", "asphalt"],
        "Concrete": ["beton", "бетон", "concrete"],
        "Gravel": ["shag'al", "гравий", "gravel", "щебень"],
        "Dirt": ["tuproq", "грунт", "dirt", "грунтовой"],
        "Paving slabs": ["plitka", "тротуарная плитка", "paving slabs", "брусчатка"],
        "Other": ["boshqa", "другой", "other"],
    },
    "document_status": {
        "Ready": [
            "tayyor", "hujjat tayyor", "готовы", "документы готовы", "ready",
        ],
        "In process": [
            "jarayonda", "в процессе", "in process",
            "hujjatlar rasmiylashtirilmoqda", "оформляются",
        ],
        "Not ready": ["tayyor emas", "не готовы", "not ready"],
        "Problematic": ["muammoli", "проблемные", "problematic"],
    },
}


class EmbeddingMatcher:
    """Semantic matcher using multilingual-e5-small embeddings via FastEmbed."""

    def __init__(self):
        """Load embedding model and pre-compute all embeddings."""
        print("Embedding modeli yuklanmoqda (multilingual-e5-small)...")
        self.model = TextEmbedding("intfloat/multilingual-e5-small")
        print("Embedding modeli yuklandi!")

        # Pre-compute embeddings
        self._build_category_index()
        self._build_enum_index()

    # ============================
    # Index Building
    # ============================

    def _embed_passages(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts as passages (for the indexed items)."""
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = list(self.model.embed(prefixed))
        return np.array(embeddings)

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a single text as a query."""
        prefixed = [f"query: {text}"]
        embeddings = list(self.model.embed(prefixed))
        return np.array(embeddings[0])

    def _build_category_index(self):
        """Pre-compute embeddings for all category synonyms."""
        self.category_texts = []
        self.category_labels = []

        for cat_name, synonyms in CATEGORY_SYNONYMS.items():
            all_terms = [cat_name.lower()] + [s.lower() for s in synonyms]
            for term in all_terms:
                self.category_texts.append(term)
                self.category_labels.append(cat_name)

        print(f"  Kategoriya indexi qurilmoqda ({len(self.category_texts)} ta term)...")
        self.category_embeddings = self._embed_passages(self.category_texts)
        # Pre-compute norms for vectorized similarity
        norms = np.linalg.norm(self.category_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.category_embeddings_normed = self.category_embeddings / norms
        print("  Kategoriya indexi tayyor!")

    def _build_enum_index(self):
        """Pre-compute embeddings for all enum field values."""
        self.enum_indices = {}

        for field_name, values_dict in ENUM_SYNONYMS.items():
            texts = []
            labels = []

            for value, synonyms in values_dict.items():
                all_terms = [value.lower()] + [s.lower() for s in synonyms]
                for term in all_terms:
                    texts.append(term)
                    labels.append(value)

            print(f"  Enum index qurilmoqda: {field_name} ({len(texts)} ta term)...")
            embeddings = self._embed_passages(texts)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings_normed = embeddings / norms

            self.enum_indices[field_name] = {
                'texts': texts,
                'labels': labels,
                'embeddings': embeddings,
                'embeddings_normed': embeddings_normed,
            }

        print("  Barcha enum indexlari tayyor!")

    # ============================
    # Matching Methods
    # ============================

    def _cosine_similarities(self, query_normed: np.ndarray, index_normed: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity between a query and all index vectors."""
        return np.dot(index_normed, query_normed)

    def embed_query(self, text: str) -> np.ndarray:
        """Public method to embed a query text. Returns normalized vector."""
        emb = self._embed_query(text)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    def detect_category(self, text: str, query_normed: np.ndarray = None) -> Tuple[str, float]:
        """Detect the best matching category for the input text.

        Args:
            text: Input text (used if query_normed is None)
            query_normed: Pre-computed normalized query embedding

        Returns:
            Tuple of (category_name, similarity_score)
        """
        if query_normed is None:
            query_normed = self.embed_query(text)

        similarities = self._cosine_similarities(query_normed, self.category_embeddings_normed)

        best_idx = int(np.argmax(similarities))
        best_category = self.category_labels[best_idx]
        best_score = float(similarities[best_idx])

        return best_category, best_score

    def match_enum(
        self,
        text: str,
        field_name: str,
        threshold: float = 0.3,
        query_normed: np.ndarray = None,
    ) -> Optional[Tuple[str, float]]:
        """Match input text to the best enum value for a given field.

        Returns:
            Tuple of (enum_value, score) or None if below threshold
        """
        if field_name not in self.enum_indices:
            return None

        if query_normed is None:
            query_normed = self.embed_query(text)

        index = self.enum_indices[field_name]
        similarities = self._cosine_similarities(query_normed, index['embeddings_normed'])

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < threshold:
            return None

        return index['labels'][best_idx], best_score

    def match_all_enums(
        self,
        text: str,
        threshold: float = 0.3,
        query_normed: np.ndarray = None,
        fields: List[str] = None,
    ) -> Dict[str, Tuple[str, float]]:
        """Match input text against multiple enum fields at once.

        Args:
            text: Input text
            threshold: Minimum similarity threshold
            query_normed: Pre-computed normalized query embedding
            fields: Specific fields to match (None = all fields)

        Returns:
            Dict of field_name -> (enum_value, score)
        """
        if query_normed is None:
            query_normed = self.embed_query(text)

        results = {}
        target_fields = fields if fields else list(self.enum_indices.keys())

        for field_name in target_fields:
            if field_name not in self.enum_indices:
                continue

            index = self.enum_indices[field_name]
            similarities = self._cosine_similarities(query_normed, index['embeddings_normed'])

            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            if best_score >= threshold:
                results[field_name] = (index['labels'][best_idx], best_score)

        return results

    def get_top_categories(
        self,
        text: str,
        top_k: int = 5,
        query_normed: np.ndarray = None,
    ) -> List[Tuple[str, float]]:
        """Get top-k matching categories with scores (unique categories)."""
        if query_normed is None:
            query_normed = self.embed_query(text)

        similarities = self._cosine_similarities(query_normed, self.category_embeddings_normed)

        # Aggregate: keep best score per category
        category_scores: Dict[str, float] = {}
        for label, score in zip(self.category_labels, similarities):
            score_f = float(score)
            if label not in category_scores or score_f > category_scores[label]:
                category_scores[label] = score_f

        sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_cats[:top_k]
