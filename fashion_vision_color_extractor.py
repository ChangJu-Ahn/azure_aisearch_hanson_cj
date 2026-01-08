#!/usr/bin/env python3
"""
Fashion Color Extractor w. Azure Computer Vision
Azure Computer Vision API(3.2 를 씁니다. 4에는 Color Scheme이 없습니다.)를 사용하여 패션 상품 이미지에서 색상 정보를 추출하고
기존 Azure AI Search 인덱스에 product_color 필드로 추가합니다.

주요 기능:
- fashion_products.csv의 image_link에서 이미지 색상 분석(dominant colors)
- Azure Computer Vision API를 통한 주요 색상 추출
- 기존 fashion-sample AI Search 인덱스에 product_color 필드 추가

사용법:
    python fashion_vision_color_extractor.py --csv fashion_products.csv
    python fashion_vision_color_extractor.py --csv fashion_products.csv --batch_size 5

필수 환경 변수 (.env 파일):
    AZURE_COMPUTER_VISION_ENDPOINT=https://your-vision-service.cognitiveservices.azure.com/
    AZURE_COMPUTER_VISION_KEY=your-vision-key
    AZURE_SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
    AZURE_SEARCH_ADMIN_KEY=your-admin-key
    FASHION_INDEX_NAME=fashion-sample
"""

import os
import sys
import pandas as pd
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
    from msrest.authentication import CognitiveServicesCredentials
except ImportError:
    print("Azure Computer Vision package not installed. Run: pip install azure-cognitiveservices-vision-computervision")
    sys.exit(1)

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import SearchableField, SearchFieldDataType
except ImportError:
    print("Azure Search packages not installed. Run: pip install azure-search-documents")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv package not installed. Run: pip install python-dotenv")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="Processing", **kwargs):
        print(f"{desc}...")
        return iterable

# Load environment variables
load_dotenv(override=True)

# Configuration
vision_endpoint = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT")
vision_key = os.getenv("AZURE_COMPUTER_VISION_KEY")
search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
fashion_index_name = os.getenv("FASHION_INDEX_NAME", "fashion-sample")


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger


logger = setup_logging()


@dataclass
class ColorInfo:
    """Color information extracted from image"""
    dominant_colors: List[str]
    accent_color: Optional[str]
    foreground_color: Optional[str]
    background_color: Optional[str]
    is_bw_image: bool = False


class FashionColorExtractor:
    """Extract colors from fashion product images using Azure Computer Vision"""
    
    def __init__(self):
        # Validate environment variables
        if not all([vision_endpoint, vision_key, search_endpoint, search_admin_key]):
            raise ValueError("Missing required environment variables. Please check your .env file.")
        
        # Initialize Computer Vision client
        self.vision_client = ComputerVisionClient(
            vision_endpoint,
            CognitiveServicesCredentials(vision_key)
        )
        
        # Initialize Search clients
        self.search_credential = AzureKeyCredential(search_admin_key)
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=fashion_index_name,
            credential=self.search_credential
        )
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=self.search_credential
        )
        
        logger.info(f"Initialized Fashion Color Extractor for index: {fashion_index_name}")
    
    def extract_colors_from_image(self, image_url: str) -> Optional[ColorInfo]:
        """Extract color information from image URL using Azure Computer Vision"""
        try:
            # Analyze image for color information
            visual_features = [VisualFeatureTypes.color]
            
            analysis = self.vision_client.analyze_image(
                image_url,
                visual_features=visual_features
            )
            
            if not analysis.color:
                logger.warning(f"No color information found for image: {image_url}")
                return None
            
            color_info = ColorInfo(
                dominant_colors=analysis.color.dominant_colors,
                accent_color=analysis.color.accent_color,
                foreground_color=analysis.color.dominant_color_foreground,
                background_color=analysis.color.dominant_color_background,
                is_bw_image=analysis.color.is_bw_img
            )
            
            return color_info
            
        except Exception as e:
            logger.error(f"Failed to extract colors from {image_url}: {e}")
            return None
    
    def format_color_string(self, color_info: ColorInfo) -> str:
        """Format color information into a clean searchable string with dominant colors only"""
        if not color_info:
            return ""
        
        # Only use dominant colors for clean, consistent results
        if not color_info.dominant_colors:
            return ""
        
        # Filter to keep only English color names (remove Korean text and color codes)
        english_colors = []
        for color in color_info.dominant_colors:
            # Skip if contains Korean characters
            if any('\u3131' <= c <= '\u3163' or '\uac00' <= c <= '\ud7a3' for c in color):
                continue
            
            # Skip if it's a color code (6 digit hex or pure numbers)
            if (len(color) == 6 and all(c in '0123456789ABCDEF' for c in color.upper())) or color.isdigit():
                continue
            
            # Keep only if it looks like an English color name
            if color.isalpha() and len(color) > 1:
                english_colors.append(color)
        
        # Remove duplicates while preserving order
        unique_colors = list(dict.fromkeys(english_colors))
        
        # Return space-separated clean color names
        return " ".join(unique_colors)
    
    def add_product_color_field_to_index(self) -> bool:
        """Add product_color field to the index schema if it doesn't exist"""
        try:
            # Get current index schema
            current_index = self.index_client.get_index(fashion_index_name)
            current_fields = {field.name: field for field in current_index.fields}
            
            # Check if product_color field already exists
            if "product_color" in current_fields:
                logger.info("product_color field already exists in index - ready for updates")
                return True
            
            logger.info("Adding product_color field to index schema...")
            
            # Add product_color field
            product_color_field = SearchableField(
                name="product_color",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
                facetable=True
            )
            
            # Update index schema
            updated_fields = list(current_index.fields) + [product_color_field]
            
            from azure.search.documents.indexes.models import SearchIndex
            updated_index = SearchIndex(
                name=current_index.name,
                fields=updated_fields,
                scoring_profiles=current_index.scoring_profiles,
                cors_options=current_index.cors_options,
                suggesters=current_index.suggesters,
                analyzers=current_index.analyzers,
                tokenizers=current_index.tokenizers,
                token_filters=current_index.token_filters,
                char_filters=current_index.char_filters,
                normalizers=current_index.normalizers,
                semantic_search=current_index.semantic_search,
                vector_search=current_index.vector_search
            )
            
            self.index_client.create_or_update_index(updated_index)
            logger.info("Successfully added product_color field to index schema")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add product_color field to index: {e}")
            return False
    
    def check_existing_color_data(self, product_ids: List[str]) -> Dict[str, bool]:
        """Check which products already have color data in the index"""
        existing_colors = {}
        
        try:
            # Query in batches to check existing color data
            batch_size = 50
            for i in range(0, len(product_ids), batch_size):
                batch_ids = product_ids[i:i+batch_size]
                id_filter = " or ".join([f"id eq '{pid}'" for pid in batch_ids])
                
                results = self.search_client.search(
                    search_text="*",
                    filter=id_filter,
                    select=["id", "product_color"],
                    top=batch_size
                )
                
                for result in results:
                    product_id = result.get('id')
                    has_color = bool(result.get('product_color'))
                    existing_colors[product_id] = has_color
                    
        except Exception as e:
            logger.warning(f"Failed to check existing color data: {e}")
            # If check fails, assume no existing data
            for pid in product_ids:
                existing_colors[pid] = False
        
        return existing_colors

    def process_csv_and_extract_colors(self, csv_path: str, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Process CSV file and extract colors from product images"""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} products from {csv_path}")
            
            processed_products = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting colors"):
                try:
                    image_url = row.get('image_link', '')
                    product_id = row.get('id', '')
                    
                    if not image_url:
                        logger.warning(f"No image URL for product {product_id}")
                        continue
                    
                    # Extract colors from image
                    color_info = self.extract_colors_from_image(image_url)
                    
                    if color_info:
                        product_color = self.format_color_string(color_info)
                        
                        processed_products.append({
                            "id": str(product_id),
                            "product_color": product_color
                        })
                    
                    # Rate limiting
                    if (idx + 1) % batch_size == 0:
                        time.sleep(1)  # Avoid rate limits
                
                except Exception as e:
                    logger.error(f"Failed to process product {product_id}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(processed_products)} products")
            return processed_products
            
        except Exception as e:
            logger.error(f"Failed to process CSV file: {e}")
            raise
    
    def update_search_index(self, color_data: List[Dict[str, Any]], batch_size: int = 10) -> bool:
        """Update search index with extracted color information"""
        try:
            logger.info(f"Updating search index with {len(color_data)} color records")
            
            total_successful = 0
            total_failed = 0
            
            # Update in batches
            for i in tqdm(range(0, len(color_data), batch_size), desc="Updating index"):
                batch = color_data[i:i+batch_size]
                
                try:
                    # Use merge_or_upload_documents to update existing documents
                    result = self.search_client.merge_or_upload_documents(documents=batch)
                    
                    # Check for failures
                    if result:
                        batch_successful = len([item for item in result if item.succeeded])
                        batch_failed = len([item for item in result if not item.succeeded])
                        
                        total_successful += batch_successful
                        total_failed += batch_failed
                        
                        # Only log if there are failures
                        if batch_failed > 0:
                            failed_items = [item for item in result if not item.succeeded]
                            logger.warning(f"Batch {i//batch_size + 1}: {batch_failed} failures out of {len(batch)}")
                            for item in failed_items[:3]:  # Show only first 3 failures to avoid log spam
                                logger.error(f"Failed: {item.key} - {item.error_message}")
                            if len(failed_items) > 3:
                                logger.error(f"... and {len(failed_items) - 3} more failures")
                
                except Exception as e:
                    logger.error(f"Batch {i//batch_size + 1} update failed: {e}")
                    total_failed += len(batch)
                    continue
            
            # Summary log
            logger.info(f"Index update completed: {total_successful} successful, {total_failed} failed")
            return total_successful > 0
            
        except Exception as e:
            logger.error(f"Failed to update search index: {e}")
            return False
    
    def run_color_extraction(self, csv_path: str, batch_size: int = 10) -> bool:
        """Complete workflow: extract colors and update index"""
        try:
            logger.info("Starting fashion color extraction workflow")
            
            # Step 1: Add product_color field to index
            if not self.add_product_color_field_to_index():
                return False
            
            # Step 2: Process CSV and extract colors
            color_data = self.process_csv_and_extract_colors(csv_path, batch_size)
            
            if not color_data:
                logger.error("No color data extracted")
                return False
            
            # Step 3: Update search index
            success = self.update_search_index(color_data, batch_size)
            
            if success:
                logger.info("Fashion color extraction completed successfully")
                return True
            else:
                logger.error("Failed to update search index")
                return False
                
        except Exception as e:
            logger.error(f"Color extraction workflow failed: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract colors from fashion product images using Azure Computer Vision"
    )
    
    parser.add_argument(
        "--csv", "-c",
        required=True,
        help="Path to CSV file containing fashion products with image_link column"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=10,
        help="Batch size for processing and updates (default: 10)"
    )
    parser.add_argument(
        "--force-reprocess", "-f",
        action="store_true",
        help="Force reprocess all products even if they already have color info"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.verbose)
    
    try:
        # Validate input file
        if not Path(args.csv).exists():
            logger.error(f"CSV file not found: {args.csv}")
            sys.exit(1)
        
        # Initialize extractor
        extractor = FashionColorExtractor()
        
        # Run color extraction
        logger.info("Starting fashion color extraction process")
        logger.info(f"Input CSV: {args.csv}")
        logger.info(f"Batch size: {args.batch_size}")
        
        success = extractor.run_color_extraction(args.csv, args.batch_size)
        
        if success:
            print(f"\n{'='*60}")
            print(f"COLOR EXTRACTION COMPLETED")
            print(f"{'='*60}")
            print(f"Input file: {args.csv}")
            print(f"Target index: {fashion_index_name}")
            print(f"Status: SUCCESS")
            print(f"{'='*60}")
            logger.info("Fashion color extraction completed successfully")
        else:
            logger.error("Fashion color extraction failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Color extraction process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
