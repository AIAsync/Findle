"""
Extraction pipeline that combines Regex, NER, and Embedding matchers.
Resolves conflicts and maps output to the target JSON schema.
"""

from typing import Dict, Any, Tuple
import re

from .regex_extractor import RegexExtractor
from .ner_extractor import NERExtractor
from .embedding_matcher import EmbeddingMatcher

class ExtractionPipeline:
    def __init__(self):
        """Initialize all extractors."""
        print("Pipeline initsializatsiya qilinmoqda...")
        self.regex = RegexExtractor()
        self.ner = NERExtractor()
        self.matcher = EmbeddingMatcher()
        print("Pipeline tayyor!")

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Main extraction function.
        Returns data formatted according to the real-estate.json schema.
        """
        # 1. Get raw extractions
        regex_data = self.regex.extract(text)
        ner_data = self.ner.extract(text)
        
        # 2. Determine category
        query_normed = self.matcher.embed_query(text)
        category, cat_score = self.matcher.detect_category(text, query_normed)
        
        # 3. Determine enum fields (global and category-specific)
        enum_matches = self.matcher.match_all_enums(
            text, 
            threshold=0.3, 
            query_normed=query_normed
        )
        
        # 4. Map to target schema format
        result = self._build_schema_output(category, text, regex_data, ner_data, enum_matches)
        return result

    def _build_schema_output(
        self, 
        category: str, 
        original_text: str,
        regex_data: Dict[str, Any], 
        ner_data: Dict[str, Any], 
        enum_matches: Dict[str, Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Build the final JSON structure according to schema."""
        
        # Base item properties
        item = {}
        
        # Property Type
        if 'property_type' in enum_matches:
            item['property_type'] = enum_matches['property_type'][0]
        else:
            item['property_type'] = "Other"
            
        # Transaction Type
        if 'transaction_type' in enum_matches:
            item['transaction_type'] = enum_matches['transaction_type'][0]
        else:
            item['transaction_type'] = "Sale" # Default
            
        # Payment Terms
        if 'payment_terms' in enum_matches:
            item['payment_terms'] = enum_matches['payment_terms'][0]
        else:
            item['payment_terms'] = "Cash"
            
        # Total Area
        if 'total_area' in regex_data:
            item['total_area'] = regex_data['total_area']
        elif 'land_area' in regex_data and category in ["Houses", "Townhouses", "Plots"]:
            item['total_area'] = regex_data['land_area']
        else:
            item['total_area'] = 0.0 # Required by schema
            
        # Geography
        if 'region' in ner_data:
            item['region'] = ner_data['region']
        else:
            item['region'] = "Unknown"
            
        if 'district' in ner_data:
            item['district'] = ner_data['district']
        else:
            item['district'] = "Unknown"
            
        if 'address' in ner_data:
            item['address'] = ner_data['address']
        else:
            item['address'] = "Unknown"
            
        if 'landmark' in ner_data:
            item['landmark'] = ner_data['landmark']
            
        # Contact
        if 'contact' in regex_data:
            item['contact'] = regex_data['contact']
        elif 'contact_phone' in ner_data:
            item['contact'] = ner_data['contact_phone']
        else:
            item['contact'] = "Unknown"
            
        # Location (dummy for now as we don't geocode)
        item['location'] = {"latitude": 0.0, "longitude": 0.0}
        
        # Specs (category specific)
        item['specs'] = self._build_specs(category, regex_data, enum_matches)
        
        # Document Status
        if 'document_status' in enum_matches:
            item['document_status'] = enum_matches['document_status'][0]
            
        # Combine into final format
        final_output = {
            "name": category,
            "items": [item]
        }
        
        # Add price if found (not in schema strictly, but useful)
        if 'price' in regex_data:
            item['price'] = regex_data['price']
            
        return final_output

    def _build_specs(
        self, 
        category: str, 
        regex_data: Dict[str, Any], 
        enum_matches: Dict[str, Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Build the 'specs' object based on category."""
        specs = {}
        
        # Enums
        enum_fields = [
            'condition', 'building_type', 'storage_type', 'office_class',
            'zoning', 'traffic', 'surface_type', 'garage_type', 'soil_quality',
            'bathroom_type', 'indoor_outdoor'
        ]
        for field in enum_fields:
            if field in enum_matches:
                specs[field] = enum_matches[field][0]
                
        # Regex numeric fields
        num_mappings = {
            'rooms': 'rooms',
            'floor': 'floor',
            'total_floors': 'total_floors',
            'year_built': 'year_built',
            'ceiling_height': 'ceiling_height',
            'land_area': 'land_area',
            'building_area': 'building_area',
            'kitchen_area': 'kitchen_area',
            'hall_area': 'hall_area',
            'distance_to_city': 'distance_to_city',
            'parking_spaces': 'parking_spaces',
            'total_spaces': 'total_spaces',
            'power_capacity': 'power_capacity',
            'stars': 'stars',
            'total_rooms': 'total_rooms',
            'neighbors': 'neighbors',
            'classrooms': 'classrooms',
            'total_beds': 'total_beds',
            'bathrooms': 'bathrooms',
            'plumbing_points': 'plumbing_points',
        }
        
        for regex_key, spec_key in num_mappings.items():
            if regex_key in regex_data:
                specs[spec_key] = regex_data[regex_key]
                
        # Boolean fields
        if 'booleans' in regex_data:
            for bool_key, bool_val in regex_data['booleans'].items():
                specs[bool_key] = bool_val
                
        # Some specific fallback logic for booleans
        if 'furniture' in regex_data.get('booleans', {}):
            specs['furnished'] = regex_data['booleans']['furniture']
            
        return specs
