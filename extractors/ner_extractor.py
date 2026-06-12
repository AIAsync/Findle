"""
NER-based data extractor using GLiNER multi model.
Zero-shot named entity recognition for real estate listings.
Supports Uzbek and Russian languages.
"""

from gliner import GLiNER
from typing import Dict, Any, List


class NERExtractor:
    """Extracts named entities from real estate text using GLiNER."""

    def __init__(self):
        """Load GLiNER multi model."""
        print("NER modeli yuklanmoqda (gliner_multi-v2.1)...")
        self.model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        print("NER modeli yuklandi!")

        # Entity labels for zero-shot NER
        self.labels = [
            "address",
            "city",
            "region",
            "district",
            "street",
            "landmark",
            "person name",
            "organization",
            "phone number",
        ]

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text and map to schema fields."""
        entities = self.model.predict_entities(
            text,
            self.labels,
            threshold=0.3
        )

        result = {}

        # Group entities by label, keep the highest-scored one per label
        entity_groups: Dict[str, Dict] = {}
        for entity in entities:
            label = entity['label']
            if label not in entity_groups or entity['score'] > entity_groups[label]['score']:
                entity_groups[label] = entity

        # Map to schema fields
        for label, entity in entity_groups.items():
            value = entity['text'].strip()

            if label == 'address':
                result['address'] = value
            elif label == 'street':
                if 'address' not in result:
                    result['address'] = value
                else:
                    result['address'] = value + ', ' + result['address']
            elif label in ('city', 'region'):
                result['region'] = value
            elif label == 'district':
                result['district'] = value
            elif label == 'landmark':
                result['landmark'] = value
            elif label == 'person name':
                result['contact_name'] = value
            elif label == 'organization':
                result['contact_org'] = value
            elif label == 'phone number':
                result['contact_phone'] = value

        # Store raw entities for debugging/meta
        result['_all_entities'] = [
            {
                'text': e['text'],
                'label': e['label'],
                'score': round(e['score'], 3)
            }
            for e in entities
        ]

        return result

    def extract_all(self, text: str) -> List[Dict[str, Any]]:
        """Extract all entity occurrences (not just best per label)."""
        entities = self.model.predict_entities(
            text,
            self.labels,
            threshold=0.3
        )
        return [
            {
                'text': e['text'],
                'label': e['label'],
                'score': round(e['score'], 3),
                'start': e.get('start'),
                'end': e.get('end'),
            }
            for e in entities
        ]
