#!/usr/bin/env python3
"""
Fashion Product Data Uploader for AI Search Index
이 유틸리티는 fashion_data_generator.py에서 생성된 패션 제품 데이터를 
fashion_index.ipynb에서 정의된 fashion-sample Azure AI Search 인덱스에 업로드합니다.

주요 기능:
- fashion_data_generator.py 출력 JSON 파일 로드
- 패션 데이터를 fashion-sample 인덱스 스키마 형식으로 변환
- Ada-002 및 Large-3 임베딩 생성
- Azure AI Search 인덱스에 배치 업로드

사용법:
    # 임베딩과 함께 업로드
    python upload_fashion_to_search.py --input fashion_products.json --create_embeddings
    
    # 임베딩 없이 업로드 (더 빠름)
    python upload_fashion_to_search.py --input fashion_products.json
    
    # 특정 인덱스에 업로드
    python upload_fashion_to_search.py --input fashion_products.json --index my-fashion-index --create_embeddings

필수 환경 변수 (.env 파일):
    AZURE_SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
    AZURE_SEARCH_ADMIN_KEY=your-admin-key
    FASHION_INDEX_NAME=fashion-sample
    AZURE_OPENAI_ENDPOINT=https://your-openai-service.openai.azure.com
    AZURE_OPENAI_KEY=your-openai-key
    AZURE_OPENAI_ADA002_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
    AZURE_OPENAI_3_LARGE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4

try:
    from openai import AzureOpenAI
except ImportError:
    print("❌ OpenAI package not installed. Run: pip install openai>=1.0.0")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("❌ python-dotenv package not installed. Run: pip install python-dotenv>=1.0.0")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("❌ tqdm package not installed. Run: pip install tqdm>=4.65.0")
    # Fallback to simple iteration if tqdm is not available
    def tqdm(iterable, desc="Processing", **kwargs):
        print(f"{desc}...")
        return iterable

# Load environment variables
load_dotenv(override=True)

# Azure Search and OpenAI configuration
search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
fashion_index_name = os.getenv("FASHION_INDEX_NAME", "fashion-sample")  # 기본값: fashion-sample
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_key = os.getenv("AZURE_OPENAI_KEY")
ada002_deployment = os.getenv("AZURE_OPENAI_ADA002_EMBEDDING_DEPLOYMENT")
large3_deployment = os.getenv("AZURE_OPENAI_3_LARGE_EMBEDDING_DEPLOYMENT")

# Azure Search imports
try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient, SearchIndexingBufferedSender
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchField, SearchFieldDataType, SimpleField, 
        SearchableField, ComplexField, SearchIndex
    )
except ImportError:
    print("Azure Search packages not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """Setup comprehensive logging"""
    log_level = logging.DEBUG if verbose else logging.WARN
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    root_logger.info("Fashion Data Uploader initialized")
    root_logger.info(f"Log level: {logging.getLevelName(log_level)}")
    
    return root_logger


logger = setup_logging()


class FashionSearchIndexUploader:
    """Upload fashion product data to Azure AI Search index (fashion-sample schema)"""
    
    def __init__(self, index_name: str = None):
        # Use environment variable if index_name not provided
        self.index_name = index_name or fashion_index_name
        
        # Validate environment variables
        if not all([search_endpoint, admin_key, openai_endpoint, openai_key]):
            raise ValueError("Missing required environment variables. Please check your .env file.")
        
        # Setup credentials
        self.search_credential = AzureKeyCredential(admin_key)
        
        # Initialize clients
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=self.index_name,
            credential=self.search_credential,
        )
        
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint, 
            credential=self.search_credential
        )
        
        # Initialize OpenAI clients for embeddings
        self.ada002_client = AzureOpenAI(
            azure_deployment=ada002_deployment,
            api_version="2024-10-21",
            azure_endpoint=openai_endpoint,
            api_key=openai_key,
        )
        
        self.large3_client = AzureOpenAI(
            azure_deployment=large3_deployment,
            api_version="2024-10-21",
            azure_endpoint=openai_endpoint,
            api_key=openai_key,
        )
        
        logger.info(f"Initialized uploader for index: {index_name}")
    
    def update_index_schema(self, sample_product: Dict[str, Any]) -> bool:
        """Update index schema to include new fields from fashion data"""
        try:
            # Get current index
            current_index = self.index_client.get_index(self.index_name)
            current_fields = {field.name: field for field in current_index.fields}
            
            # Define new fashion fields that should be added
            new_fields = []
            
            # Check and add new fields based on sample product
            fashion_field_mappings = {
                'productCode': SearchableField(name="productCode", type=SearchFieldDataType.String, 
                                             searchable=True, filterable=True, facetable=True),
                'style': SearchableField(name="style", type=SearchFieldDataType.String, 
                                       searchable=True, filterable=True, facetable=True),
                'color': SearchableField(name="color", type=SearchFieldDataType.String, 
                                       searchable=True, filterable=True, facetable=True),
                'material': SearchableField(name="material", type=SearchFieldDataType.String, 
                                          searchable=True, filterable=True, facetable=True),
                'targetGender': SimpleField(name="targetGender", type=SearchFieldDataType.String, 
                                          filterable=True, facetable=True),
                'targetAge': SearchableField(name="targetAge", type=SearchFieldDataType.String, 
                                           searchable=True, filterable=True, facetable=True),
                'season': SimpleField(name="season", type=SearchFieldDataType.String, 
                                    filterable=True, facetable=True),
                'features': SearchableField(name="features", type=SearchFieldDataType.String, 
                                          searchable=True),
                'careInstructions': SearchableField(name="careInstructions", type=SearchFieldDataType.String, 
                                                  searchable=True),
                'styleTags': SearchableField(name="styleTags", type=SearchFieldDataType.String, 
                                           searchable=True, filterable=True, facetable=True),
                'occasionTags': SearchableField(name="occasionTags", type=SearchFieldDataType.String, 
                                              searchable=True, filterable=True, facetable=True),
                'seasonTags': SearchableField(name="seasonTags", type=SearchFieldDataType.String, 
                                            searchable=True, filterable=True, facetable=True),
                'ageTags': SearchableField(name="ageTags", type=SearchFieldDataType.String, 
                                         searchable=True, filterable=True, facetable=True),
                'genderTags': SearchableField(name="genderTags", type=SearchFieldDataType.String, 
                                            searchable=True, filterable=True, facetable=True),
                'sizeRange': SearchableField(name="sizeRange", type=SearchFieldDataType.String, 
                                           searchable=True, filterable=True, facetable=True),
                'brandPositioning': SearchableField(name="brandPositioning", type=SearchFieldDataType.String, 
                                                  searchable=True),
                'rating': SimpleField(name="rating", type=SearchFieldDataType.Double, 
                                    filterable=True, sortable=True, facetable=True),
                'reviewCount': SimpleField(name="reviewCount", type=SearchFieldDataType.Int32, 
                                         filterable=True, sortable=True, facetable=True),
            }
            
            # Check which fields need to be added
            fields_to_add = []
            for field_name, field_def in fashion_field_mappings.items():
                if field_name not in current_fields:
                    fields_to_add.append(field_def)
            
            if not fields_to_add:
                logger.info("Index schema already contains all required fields")
                return True
            
            # Add new fields to current index
            updated_fields = list(current_index.fields) + fields_to_add
            
            # Create updated index
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
                semantic_search=current_index.semantic_search
            )
            
            # Update the index
            self.index_client.create_or_update_index(updated_index)
            
            logger.info(f"✅ Successfully added {len(fields_to_add)} new fields to index schema:")
            for field in fields_to_add:
                logger.info(f"   - {field.name} ({field.type})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update index schema: {e}")
            return False
    
    def load_fashion_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load fashion product data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle new JSON structure with 'products' key
            if isinstance(data, dict) and 'products' in data:
                products = data['products']
                logger.info(f"📁 Loaded {len(products)} fashion products from metadata structure: {file_path}")
            elif isinstance(data, list):
                products = data
                logger.info(f"📁 Loaded {len(products)} fashion products from list structure: {file_path}")
            else:
                logger.error(f"❌ Unexpected data structure in {file_path}")
                return []
            
            return products
            
        except Exception as e:
            logger.error(f"❌ Failed to load data from {file_path}: {e}")
            raise
    
    def get_embedding_ada002(self, text: str) -> List[float]:
        """Generate Ada-002 embedding"""
        try:
            response = self.ada002_client.embeddings.create(
                input=[text], 
                model=ada002_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Ada-002 embedding generation failed: {e}")
            return None

    def get_embedding_large3(self, text: str) -> List[float]:
        """Generate Large-3 embedding"""
        try:
            response = self.large3_client.embeddings.create(
                input=[text], 
                model=large3_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Large-3 embedding generation failed: {e}")
            return None
    
    def get_existing_document_by_id(self, document_id: str) -> Dict[str, Any]:
        """Get existing document from index by document ID"""
        try:
            # Search for existing document using document ID
            results = self.search_client.search(
                search_text="*",
                filter=f"id eq '{document_id}'",
                top=1
            )
            
            for result in results:
                return dict(result)
            return None
            
        except Exception as e:
            logger.debug(f"Could not find existing document for ID {document_id}: {e}")
            return None
    
    def merge_with_existing_document(self, existing_doc: Dict[str, Any], fashion_product: Dict[str, Any]) -> Dict[str, Any]:
        """Merge existing shopping document with new fashion data"""
        
        # Start with existing document to preserve all original fields
        merged_doc = existing_doc.copy()
        
        # Update/add fashion-specific fields from the new data
        fashion_fields = [
            'style', 'color', 'material', 'targetGender', 'targetAge', 'season',
            'features', 'careInstructions', 'styleTags', 'occasionTags', 
            'seasonTags', 'ageTags', 'genderTags', 'sizeRange', 
            'brandPositioning', 'rating', 'reviewCount'
        ]
        
        # Always preserve/update productCode if provided
        if 'productCode' in fashion_product:
            merged_doc['productCode'] = fashion_product['productCode']
        
        # Update only the new fashion fields, preserve all existing fields
        for field in fashion_fields:
            if field in fashion_product:
                value = fashion_product[field]
                # Only update if value is meaningful
                if value and value != '정보 없음' and value != 'Unknown':
                    # Convert arrays to strings for Azure Search
                    if isinstance(value, list):
                        merged_doc[field] = ' '.join(value)
                    else:
                        merged_doc[field] = value
        
        # Update main_text to include new fashion information
        if any(field in fashion_product for field in fashion_fields):
            main_text_parts = [
                merged_doc.get('title', ''),
                merged_doc.get('brand', ''),
                merged_doc.get('category', ''),
                fashion_product.get('description', ''),
                ' '.join(fashion_product.get('styleTags', [])),
                ' '.join(fashion_product.get('occasionTags', [])),
                ' '.join(fashion_product.get('ageTags', [])),
                ' '.join(fashion_product.get('genderTags', [])),
                fashion_product.get('material', ''),
                fashion_product.get('style', '')
            ]
            merged_doc['main_text'] = ' '.join([part for part in main_text_parts if part and part != '정보 없음'])
            
            # Update keyword field
            all_tags = (
                fashion_product.get('styleTags', []) +
                fashion_product.get('occasionTags', []) +
                fashion_product.get('ageTags', []) +
                fashion_product.get('genderTags', []) +
                fashion_product.get('seasonTags', [])
            )
            if all_tags:
                existing_keyword = merged_doc.get('keyword', '')
                new_keywords = ' '.join(all_tags)
                merged_doc['keyword'] = f"{existing_keyword} {new_keywords}".strip()
        
        return merged_doc

    def get_price_range_code(self, price: float) -> str:
        """Convert price to price range code for shopping-sample index"""
        try:
            price = int(price)
            
            # Price range mapping similar to index.ipynb
            price_ranges = [
                (10000, "1만원미만"),
                (20000, "1만원대"),
                (30000, "2만원대"), 
                (40000, "3만원대"),
                (50000, "4만원대"),
                (60000, "5만원대"),
                (70000, "6만원대"),
                (80000, "7만원대"),
                (90000, "8만원대"),
                (100000, "9만원대"),
                (200000, "10만원대"),
                (300000, "20만원대"),
                (400000, "30만원대"),
                (500000, "40만원대"),
                (600000, "50만원대"),
                (700000, "60만원대"),
                (800000, "70만원대"),
                (900000, "80만원대"),
                (1000000, "90만원대")
            ]
            
            for threshold, label in price_ranges:
                if price < threshold:
                    return label
            
            return "100만원이상"
            
        except (ValueError, TypeError):
            return "가격미상"
    
    def convert_fashion_to_shopping_format(self, fashion_products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert fashion product data to fashion-sample index format, merging with existing docs"""
        logger.info("🔄 Converting fashion products to fashion-sample format...")
        
        converted_products = []
        updated_count = 0
        
        for product in tqdm(fashion_products, desc="Converting and merging products"):
            try:
                # Use the original ID from the fashion data for mapping
                document_id = product.get('id', '')
                product_code = product.get('productCode', '')
                
                # Check if document already exists using ID
                existing_doc = self.get_existing_document_by_id(document_id)
                
                if existing_doc:
                    # Merge with existing document (upsert only)
                    merged_product = self.merge_with_existing_document(existing_doc, product)
                    converted_products.append(merged_product)
                    updated_count += 1
                    logger.debug(f"   • Updated existing ID: {document_id} (ProductCode: {product_code})")
                else:
                    # Skip if document doesn't exist - only perform upserts
                    logger.warning(f"   • Skipped - Document not found for ID: {document_id} (ProductCode: {product_code})")
                    continue
                
            except Exception as e:
                logger.error(f"❌ Failed to convert product {product.get('productCode', 'unknown')}: {e}")
                continue
        
        logger.info(f"Converted {len(converted_products)} products (0 new, {updated_count} updated)")
        return converted_products
    
    def generate_embeddings_for_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for fashion-sample format products"""
        logger.info("🔄 Generating embeddings for fashion products...")
        
        products_with_embeddings = []
        
        for product in tqdm(products, desc="Generating embeddings"):
            try:
                main_text = product.get("main_text", "")
                
                # Generate Ada-002 embedding (1536 dimensions)
                ada002_embedding = self.get_embedding_ada002(main_text)
                if ada002_embedding is None:
                    logger.warning(f"Failed to generate Ada-002 embedding for product {product.get('id')}")
                    continue
                
                # Generate Large-3 embedding (3072 dimensions)
                large3_embedding = self.get_embedding_large3(main_text)
                if large3_embedding is None:
                    logger.warning(f"Failed to generate Large-3 embedding for product {product.get('id')}")
                    continue
                
                # Add embeddings to product
                product["main_text_vector"] = ada002_embedding
                product["main_text_vector_3"] = large3_embedding
                
                products_with_embeddings.append(product)
                
            except Exception as e:
                logger.error(f"❌ Failed to generate embeddings for product {product.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"✅ Generated embeddings for {len(products_with_embeddings)} products")
        return products_with_embeddings
    
    def verify_index_exists(self) -> bool:
        """Verify that the target index exists"""
        try:
            index = self.index_client.get_index(self.index_name)
            logger.info(f"✅ Index '{self.index_name}' exists and is ready")
            return True
        except Exception as e:
            logger.error(f"❌ Index '{self.index_name}' not found: {e}")
            logger.error("Please run fashion_index.ipynb first to create the fashion-sample index")
            return False
    
    def upload_to_index(self, products: List[Dict[str, Any]], batch_size: int = 10) -> bool:
        """Upload/update products to search index using merge operation"""
        try:
            logger.info(f"📤 Uploading/updating {len(products)} products to index: {self.index_name}")
            
            # Use direct client calls for merge operations
            failed_batches = 0
            for i in tqdm(range(0, len(products), batch_size), desc="Uploading batches"):
                batch = products[i:i+batch_size]
                try:
                    # Use merge_or_upload_documents for upsert behavior
                    result = self.search_client.merge_or_upload_documents(documents=batch)
                    
                    # Check for failures in the result
                    if result:
                        failed_items = [item for item in result if not item.succeeded]
                        if failed_items:
                            logger.error(f"❌ Batch {i//batch_size + 1} had {len(failed_items)} failures")
                            for item in failed_items:
                                logger.error(f"   • Failed: {item.key} - {item.error_message}")
                            failed_batches += 1
                        else:
                            logger.debug(f"✅ Batch {i//batch_size + 1} uploaded successfully")
                    else:
                        logger.debug(f"✅ Batch {i//batch_size + 1} uploaded successfully")
                except Exception as e:
                    logger.error(f"❌ Batch {i//batch_size + 1} upload failed: {e}")
                    failed_batches += 1
                    continue
            
            if failed_batches > 0:
                logger.warning(f"⚠️ {failed_batches} batches failed to upload completely")
                return False
                
            logger.info("✅ Successfully uploaded/updated products in search index")
            
            # Verify upload with a test search
            import time
            time.sleep(3)  # Wait for indexing
            
            test_results = self.search_client.search(search_text="*", top=1, include_total_count=True)
            total_count = test_results.get_count()
            logger.info(f"✅ Index verification - Total documents: {total_count}")
            
            # Test search for uploaded fashion products
            fashion_results = self.search_client.search(search_text="패션", top=3)
            found_fashion_products = 0
            for result in fashion_results:
                found_fashion_products += 1
                logger.info(f"   • Found: {result.get('title', 'Unknown')} ({result.get('brand', 'Unknown')})")
            
            if found_fashion_products > 0:
                logger.info(f"✅ Successfully found {found_fashion_products} fashion products in index")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload to search index: {e}")
            return False
    
    def process_and_upload(self, file_path: str, create_embeddings: bool = True, batch_size: int = 10) -> bool:
        """Complete workflow: load, convert, embed, and upload fashion data"""
        try:
            # Verify index exists
            if not self.verify_index_exists():
                return False
            
            # Load fashion product data
            fashion_products = self.load_fashion_data(file_path)
            
            # Update index schema with new fields (if needed)
            if fashion_products:
                logger.info("🔧 Checking and updating index schema...")
                if not self.update_index_schema(fashion_products[0]):
                    logger.warning("⚠️ Schema update failed, continuing with existing schema")
            
            # Convert to shopping-sample format
            shopping_products = self.convert_fashion_to_shopping_format(fashion_products)
            
            # Generate embeddings if requested
            if create_embeddings:
                shopping_products = self.generate_embeddings_for_products(shopping_products)
            
            # Upload to index
            success = self.upload_to_index(shopping_products, batch_size)
            
            if success:
                logger.info(f"🎉 Successfully processed and uploaded {len(shopping_products)} fashion products")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Process and upload failed: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Upload fashion product data to Azure AI Search index (fashion-sample)"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file containing fashion product data"
    )
    parser.add_argument(
        "--index", "-idx",
        default=None,  # Will use FASHION_INDEX_NAME from environment
        help=f"Azure AI Search index name (default: {fashion_index_name})"
    )
    parser.add_argument(
        "--create_embeddings", "-e",
        action="store_true",
        help="Generate embeddings for vector fields"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=10,
        help="Batch size for uploads (default: 10)"
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
        # Validate input file exists
        if not Path(args.input).exists():
            logger.error(f"❌ Input file not found: {args.input}")
            sys.exit(1)
        
        # Initialize uploader
        uploader = FashionSearchIndexUploader(args.index)
        
        # Process and upload
        logger.info(f"🚀 Starting fashion data upload process")
        logger.info(f"   • Input file: {args.input}")
        logger.info(f"   • Index name: {args.index}")
        logger.info(f"   • Create embeddings: {args.create_embeddings}")
        logger.info(f"   • Batch size: {args.batch_size}")
        
        success = uploader.process_and_upload(
            args.input, 
            args.create_embeddings, 
            args.batch_size
        )
        
        if success:
            print(f"\n{'='*60}")
            print(f"🎉 FASHION UPLOAD SUMMARY")
            print(f"{'='*60}")
            print(f"📁 Input file: {args.input}")
            print(f"🏪 Target index: {args.index}")
            print(f"🔄 Embeddings: {'Generated' if args.create_embeddings else 'Skipped'}")
            print(f"✅ Status: SUCCESS")
            print(f"{'='*60}")
            logger.info("✅ Fashion data upload completed successfully!")
        else:
            logger.error("❌ Fashion data upload failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Upload process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
