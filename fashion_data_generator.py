#!/usr/bin/env python3
"""
Fashion Product Data Generator w. Grounding with Bing Search

제품정보에서 메타데이터를 AI로 생성합니다. 
"""

import os
import asyncio
import json
import logging
import argparse
import sys
import re
import random
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure imports
from openai import AsyncAzureOpenAI
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import BingGroundingTool
from azure.identity import DefaultAzureCredential

# Setup logging
def setup_logging(verbose: bool = False):
    """Configure logging with colored output"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Suppress Azure SDK HTTP request logging
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("azure.ai.projects").setLevel(logging.WARNING)
    logging.getLogger("azure.ai.agents").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

logger = logging.getLogger(__name__)

def extract_model_code(product_title: str) -> str:
    """
    Extract model code from product title
    Examples:
    - "[창주스토어 더현대대구][CC콜렉트] C252KSK034 슬림 골지 카라 풀오버" → "C252KSK034"
    - "오일릴리 패턴 블라우스-OWESGBL020-02" → "OWESGBL020-02"
    - "[창주스토어 신촌점][정호진] 자켓형블라우스 (JG2B332P)" → "JG2B332P"
    """
    # Pattern 1: Text in parentheses (JG2B332P)
    parentheses_match = re.search(r'\(([A-Z0-9]+[A-Z0-9\-]*)\)', product_title)
    if parentheses_match:
        return parentheses_match.group(1)
    
    # Pattern 2: After brand] space and before space (C252KSK034)
    bracket_pattern = re.search(r'\]\s+([A-Z0-9]+[A-Z0-9\-]*)\s+', product_title)
    if bracket_pattern:
        return bracket_pattern.group(1)
    
    # Pattern 3: After dash (-OWESGBL020-02)
    dash_pattern = re.search(r'-([A-Z0-9]+[A-Z0-9\-]*)-?[0-9]*\s', product_title)
    if dash_pattern:
        return dash_pattern.group(1)
    
    # Pattern 4: Generic alphanumeric code
    generic_pattern = re.search(r'([A-Z][A-Z0-9]{4,}[A-Z0-9\-]*)', product_title)
    if generic_pattern:
        return generic_pattern.group(1)
    
    # If no pattern found, return original title
    return product_title

@dataclass
class FashionProductData:
    """Enhanced fashion product data structure"""
    id: str  # 원본 인덱스 id (CSV의 id 필드)
    productCode: str
    brandName: str
    productName: str
    price: float
    category1: str
    category2: str
    style: str = ""
    color: str = ""
    material: str = ""
    targetGender: str = "남녀공용"
    targetAge: str = "20-40대"
    season: str = "사계절"
    description: str = ""
    features: List[str] = None
    careInstructions: str = ""
    styleTags: List[str] = None
    occasionTags: List[str] = None
    seasonTags: List[str] = None
    ageTags: List[str] = None
    genderTags: List[str] = None
    sizeRange: str = ""
    brandPositioning: str = ""
    rating: float = 4.0
    reviewCount: int = 0
    
    def __post_init__(self):
        if self.features is None:
            self.features = []
        if self.styleTags is None:
            self.styleTags = []
        if self.occasionTags is None:
            self.occasionTags = []
        if self.seasonTags is None:
            self.seasonTags = []
        if self.ageTags is None:
            self.ageTags = []
        if self.genderTags is None:
            self.genderTags = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,  # 원본 인덱스 id 포함
            "productCode": self.productCode,
            "brandName": self.brandName,
            "productName": self.productName,
            "price": self.price,
            "category1": self.category1,
            "category2": self.category2,
            "style": self.style,
            "color": self.color,
            "material": self.material,
            "targetGender": self.targetGender,
            "targetAge": self.targetAge,
            "season": self.season,
            "description": self.description,
            "features": self.features,
            "careInstructions": self.careInstructions,
            "styleTags": self.styleTags,
            "occasionTags": self.occasionTags,
            "seasonTags": self.seasonTags,
            "ageTags": self.ageTags,
            "genderTags": self.genderTags,
            "sizeRange": self.sizeRange,
            "brandPositioning": self.brandPositioning,
            "rating": self.rating,
            "reviewCount": self.reviewCount
        }

class FashionDataGenerator:
    """Main class for generating fashion product data with Azure AI Foundry grounding"""
    
    def __init__(self, language: str = "ko"):
        """Initialize the generator with Azure AI Foundry configuration"""
        self.language = language
        
        # Load environment variables
        self.api_key = os.getenv("AZURE_OPENAI_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.model_deployment_name = os.getenv("MODEL_DEPLOYMENT_NAME")
        self.project_endpoint = os.getenv("PROJECT_ENDPOINT")  # Changed to endpoint-based
        self.bing_connection_name = os.getenv("BING_CONNECTION_NAME")
        
        # Shared agent for all product searches (to avoid recreating)
        self._shared_agent = None
        
        if not all([self.api_key, self.endpoint, self.model_deployment_name]):
            raise ValueError("Missing required environment variables: AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, MODEL_DEPLOYMENT_NAME")
        
        if not all([self.project_endpoint, self.bing_connection_name]):
            logger.warning("Azure AI Foundry configuration missing. Grounding with Bing Search will be disabled.")
            self.project_client = None
        else:
            try:
                # Initialize Azure AI Foundry client
                self.project_client = AIProjectClient(
                    endpoint=self.project_endpoint,
                    credential=DefaultAzureCredential()
                )
                logger.info(" Azure AI Foundry client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure AI Foundry client: {e}")
                self.project_client = None
        
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=self.endpoint
        )
        
        logger.info("Fashion Data Generator initialized")

    def load_products_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load products from CSV file"""
        try:
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found: {csv_path}")
                return []
            
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} products from {csv_path}")
            
            # Convert DataFrame to list of dictionaries
            products = []
            for _, row in df.iterrows():
                try:
                    title = str(row.get("title", "Unknown Product"))
                    model_code = extract_model_code(title)  
                    
                    product = {
                        "id": str(row.get("id")),  # 원본 인덱스 id 보존
                        "brandName": str(row.get("brand", "Unknown")),
                        "productName": title,
                        "productCode": model_code, 
                        "price": float(row.get("normal_price", 0))
                    }
                    products.append(product)
                except Exception as e:
                    logger.warning(f"Error processing row: {e}")
                    continue
            
            logger.info(f" Successfully processed {len(products)} products")
            return products
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return []

    async def get_or_create_shared_agent(self) -> Any:
        """Get or create a shared agent for all product searches"""
        if self._shared_agent is not None:
            return self._shared_agent
            
        if not self.project_client or not self.bing_connection_name:
            logger.warning("Azure AI Foundry client or Bing connection not configured.")
            return None
        
        try:
            # 1. Retrieve Bing connection from AI Foundry project
            connection_name = self.bing_connection_name.split('/')[-1]
            logger.debug(f"Using connection name: {connection_name}")
            
            # Get the connection
            bing_connection = self.project_client.connections.get(connection_name)
            conn_id = bing_connection.id
            logger.info(f" Bing Connection ID: {conn_id}")
            
            # 2. Initialize Bing grounding tool
            bing_tool = BingGroundingTool(connection_id=conn_id)
            
            # 3. Create a shared agent that can search with Bing
            self._shared_agent = self.project_client.agents.create_agent(
                model=self.model_deployment_name,
                name="fashion-analyzer-shared",
                instructions="""너는 패션 제품 정보 분석 전문가야. 주어진 패션 제품에 대해 웹에서 제품 정보를 검색하여 상세한 제품 정보를 제공해.

                다음 정보를 포함하여 상세히 분석해줘:
                1. 제품의 스타일과 디자인 특징
                2. 소재와 품질 정보
                3. 타겟 고객층과 착용 상황
                4. 색상과 사이즈 옵션
                5. 브랜드 특성과 위치
                6. 가격대와 품질 수준
                7. 사용자 후기나 평가

                웹에서 찾은 최신 정보를 바탕으로 정확하고 상세한 분석을 한국어로 제공해줘.""",
                tools=bing_tool.definitions,
                headers={"x-ms-enable-preview": "true"},
            )
            
            logger.info(f"Created shared Bing-grounded agent, ID: {self._shared_agent.id}")
            return self._shared_agent
            
        except Exception as e:
            logger.error(f"Failed to create shared agent: {e}")
            return None

    async def search_fashion_product_with_grounding(self, brand_name: str, product_code: str, product_name: str) -> str:
        """Search for detailed information using Azure AI Foundry Agents with Bing Grounding"""
        
        if not self.project_client or not self.bing_connection_name:
            logger.warning("Azure AI Foundry client or Bing connection not configured. Skipping grounding search.")
            return ""
        
        try:
            logger.info(f"🔍 Azure AI Foundry grounding search for: {brand_name} {product_code}")
            
            # Get or create shared agent
            agent = await self.get_or_create_shared_agent()
            if not agent:
                logger.warning("Failed to get shared agent")
                return ""
            
            # Create a thread for communication
            thread = self.project_client.agents.threads.create()
            logger.debug(f"Created thread: {thread.id}")
            
            # Add a message to the thread
            query = f"""
            '{brand_name}' 브랜드의 '{product_code}' (모델명) 패션 제품에 대한 상세 정보를 웹에서 검색하여 분석해주세요. 
            제품의 특징, 스타일, 소재, 타겟 고객, 브랜드 포지셔닝 등을 포함해서 종합적으로 분석해주세요.
            """

            message = self.project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
            logger.debug(f"Created message: {message.id}")
            
            # Create and run agent asynchronously with exponential backoff
            max_retries = 3
            base_delay = 1.0
            
            for retry_count in range(max_retries + 1):
                try:
                    run = self.project_client.agents.runs.create(
                        thread_id=thread.id,
                        agent_id=agent.id
                    )
                    logger.debug(f"Created run: {run.id}")
                    break
                except Exception as e:
                    if "rate_limit_exceeded" in str(e) or "429" in str(e):
                        if retry_count < max_retries:
                            # Exponential backoff with jitter
                            wait_time = base_delay * (2 ** retry_count) + random.uniform(0, 1)
                            logger.warning(f"Rate limit hit, backing off {wait_time:.1f}s (attempt {retry_count + 1}/{max_retries + 1})")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"Max retries exceeded for run creation after rate limits")
                            return ""
                    else:
                        logger.error(f"Failed to create run: {e}")
                        return ""
            
            # Poll for completion asynchronously with exponential backoff (429 대비용..)
            poll_retry_count = 0
            max_poll_retries = 5
            
            while run.status in ["queued", "in_progress", "requires_action"]:
                await asyncio.sleep(1.0)
                try:
                    run = self.project_client.agents.runs.get(
                        thread_id=thread.id,
                        run_id=run.id
                    )
                    logger.debug(f"Run status: {run.status}")
                    poll_retry_count = 0  # Reset on successful call
                except Exception as e:
                    if "rate_limit_exceeded" in str(e) or "429" in str(e):
                        if poll_retry_count < max_poll_retries:
                            # Exponential backoff for polling
                            wait_time = 2.0 * (2 ** poll_retry_count) + random.uniform(0, 1)
                            logger.warning(f"Rate limit during polling, backing off {wait_time:.1f}s")
                            await asyncio.sleep(wait_time)
                            poll_retry_count += 1
                        else:
                            logger.error("Max polling retries exceeded")
                            break
                    else:
                        logger.error(f"Error checking run status: {e}")
                        break
            
            logger.debug(f"Run finished with status: {run.status}")
            
            if run.status == "failed":
                logger.error(f"Run failed: {run.last_error}")
                return ""
            
            # Fetch all messages to get the response
            messages = self.project_client.agents.messages.list(thread_id=thread.id)
            
            # Find the assistant's response
            assistant_content = ""
            for msg in messages:
                if msg.role == "assistant":
                    # Extract text content from the message
                    if msg.content:
                        last_content = msg.content[-1]
                        if hasattr(last_content, "text"):
                            assistant_content += last_content.text.value + "\n"
            
            # Clean up thread (but keep agent for reuse)
            try:
                self.project_client.agents.delete_thread(thread.id)
                logger.debug("Deleted thread")
            except Exception as e:
                logger.debug(f"Failed to delete thread: {e}")
            
            if assistant_content.strip():
                logger.info(f" Found grounding content: {len(assistant_content)} characters")
                return assistant_content.strip()
            else:
                logger.warning("No content returned from grounding search")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Azure AI Foundry grounding search failed for {brand_name} {product_name}: {e}")
            return ""

    async def extract_fashion_product_info(self, content: str, brand_name: str, product_name: str, product_code: str, original_price: float) -> Optional[Dict[str, Any]]:
        """Extract fashion product information using AI with structured output"""
        
        try:
            messages = [
                {
                    "role": "system", 
                    "content": f"""너는 창주스토어의 패션 제품 정보 분석 전문가야. 주어진 웹 콘텐츠(이미지 포함)를 바탕으로 특정 패션 제품의 상세 정보를 추출하여 정확한 메타데이터를 생성해.

                    **분석 대상 제품:**
                    - 브랜드: {brand_name}
                    - 제품명: {product_name}
                    - 제품코드: {product_code}
                    - 기존 가격: {original_price}

                    웹 콘텐츠에서 이 제품과 관련된 정보를 찾아서 아래 JSON 형식으로 정확히 추출해줘:

                    {{
                    "brand": "브랜드명",
                    "product_name": "정확한 제품명",
                    "category": "대분류",
                    "subcategory": "소분류", 
                    "style": "스타일 (예: 캐주얼, 포멀, 스포츠)",
                    "material": "소재 정보",
                    "color": "색상 정보",
                    "size_range": "사이즈 범위",
                    "target_gender": "타겟 성별",
                    "target_age": "타겟 연령대",
                    "season": "계절 정보(예: 봄, 여름, 가을, 겨울, 사계절, 간절기, 봄여름, 가을겨울)",
                    "price": 가격정보(숫자),
                    "description": "제품 설명",
                    "features": ["특징1", "특징2"],
                    "care_instructions": "관리 방법",
                    "brand_positioning": "브랜드 포지셔닝",
                    "style_tags": ["태그1", "태그2"]
                    }}"""
                },
                {
                    "role": "user", 
                    "content": f"다음 웹 콘텐츠에서 '{brand_name}' 브랜드의 '{product_name}' 제품 정보를 분석해주세요:\n\n{content}"
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model_deployment_name,
                messages=messages,
                response_format={ "type": "json_object" },
                temperature=0.4,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content
            logger.debug(f"AI response: {result_text[:200]}...")
            
            # Parse JSON response
            result = json.loads(result_text)
            
            # Validate and normalize the response
            product_info = {
                "product_code": product_code,
                "brand": result.get("brand", brand_name),
                "product_name": result.get("product_name", product_name),
                "category": result.get("category", "패션"),
                "subcategory": result.get("subcategory", "의류"),
                "style": result.get("style", "캐주얼"),
                "material": result.get("material", ""),
                "color": result.get("color", ""),
                "size_range": result.get("size_range", ""),
                "target_gender": result.get("target_gender", "남녀공용"),
                "target_age": result.get("target_age", "20-40대"),
                "season": result.get("season", "사계절"),
                "price": result.get("price", original_price),
                "description": result.get("description", ""),
                "features": result.get("features", []),
                "care_instructions": result.get("care_instructions", ""),
                "brand_positioning": result.get("brand_positioning", ""),
                "style_tags": result.get("style_tags", [])
            }
            
            logger.info(f" Extracted product info for {brand_name} {product_name}")
            return product_info
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting product info: {e}")
            return None

    async def generate_fashion_product_data(self, count: int, csv_path: str = "fashion_products.csv") -> List[FashionProductData]:
        """Generate fashion product data from CSV file with parallel processing"""
        
        # Load products from CSV
        csv_products = self.load_products_from_csv(csv_path)
        
        if not csv_products:
            logger.warning(f"No products loaded from CSV.")
            return []
        
        total_available = len(csv_products)
        process_count = min(count, total_available)
        
        if count >= total_available:
            logger.info(f"Processing all {process_count} fashion products from CSV: {csv_path}")
        else:
            logger.info(f"Processing {process_count} out of {total_available} fashion products from CSV: {csv_path}")

        # Create semaphore to limit concurrent processing to 10
        semaphore = asyncio.Semaphore(10)
        
        # Create tasks for parallel processing
        tasks = []
        for i in range(process_count):
            csv_product = csv_products[i % total_available]
            task = self.process_single_product(semaphore, csv_product, i + 1, process_count)
            tasks.append(task)
        
        logger.info(f"� Starting parallel processing with max 10 concurrent tasks...")
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        products = []
        failed_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Task {i+1} failed: {result}")
                failed_count += 1
            elif result is not None:
                products.append(result)
        
        logger.info(f" Parallel processing completed: {len(products)} successful, {failed_count} failed")
        logger.info(f"🎉 Generated {len(products)} fashion products with Azure AI Foundry grounding!")
        return products

    async def process_single_product(self, semaphore: asyncio.Semaphore, csv_product: Dict[str, Any], 
                                   current_num: int, total_num: int) -> Optional[FashionProductData]:
        """Process a single product with semaphore control"""
        async with semaphore:
            try:
                original_id = csv_product["id"]  # CSV의 원본 id 가져오기
                brand_name = csv_product["brandName"]
                product_name = csv_product["productName"]
                product_code = csv_product["productCode"]
                original_price = csv_product["price"]
                
                logger.info(f"� Processing product {current_num}/{total_num}: {brand_name} - {product_name}")
                
                # Search for product information using Grounding with Bing Search
                grounding_content = await self.search_fashion_product_with_grounding(
                    brand_name, product_code, product_name
                )
                
                if not grounding_content or len(grounding_content) < 50:
                    logger.warning(f"No grounding content for {brand_name} {product_name}")
                    # Create basic product without AI enhancement
                    product = FashionProductData(
                        id=original_id,
                        productCode=product_code,
                        brandName=brand_name,
                        productName=product_name,
                        price=original_price,
                        category1="패션",
                        category2="의류"
                    )
                    return product
                
                # Extract product information using grounding content
                product_info = await self.extract_fashion_product_info(
                    grounding_content, brand_name, product_name, product_code, original_price
                )
                
                if not product_info:
                    logger.warning(f"Failed to extract product info for {brand_name} {product_name}")
                    # Create basic product without AI enhancement
                    product = FashionProductData(
                        id=original_id,
                        productCode=product_code,
                        brandName=brand_name,
                        productName=product_name,
                        price=original_price,
                        category1="패션",
                        category2="의류"
                    )
                    return product
                
                # Create enhanced product with AI-generated metadata
                enhanced_product = FashionProductData(
                    id=original_id,
                    productCode=product_code,
                    brandName=product_info.get("brand", brand_name),
                    productName=product_info.get("product_name", product_name),
                    price=product_info.get("price", original_price),
                    category1=product_info.get("category", "패션"),
                    category2=product_info.get("subcategory", "의류"),
                    style=product_info.get("style", "캐주얼"),
                    color=product_info.get("color", ""),
                    material=product_info.get("material", ""),
                    targetGender=product_info.get("target_gender", "남녀공용"),
                    targetAge=product_info.get("target_age", "20-40대"),
                    season=product_info.get("season", "사계절"),
                    description=product_info.get("description", ""),
                    features=product_info.get("features", []),
                    careInstructions=product_info.get("care_instructions", ""),
                    styleTags=product_info.get("style_tags", []),
                    sizeRange=product_info.get("size_range", ""),
                    brandPositioning=product_info.get("brand_positioning", ""),
                    rating=4.2,  # Default rating
                    reviewCount=50   # Default review count
                )
                
                logger.info(f" Enhanced product {current_num}: {enhanced_product.brandName} - {enhanced_product.productName}")
                
                # Small delay to avoid overwhelming APIs
                await asyncio.sleep(0.1)
                
                return enhanced_product
                
            except Exception as e:
                logger.error(f"Error processing product {current_num}: {e}")
                # Create basic product as fallback
                product = FashionProductData(
                    id=csv_product["id"],
                    productCode=csv_product["productCode"],
                    brandName=csv_product["brandName"],
                    productName=csv_product["productName"],
                    price=csv_product["price"],
                    category1="패션",
                    category2="의류"
                )
                return product

    async def save_products_to_file(self, products: List[FashionProductData], output_path: str) -> None:
        """Save generated products to JSON file"""
        try:
            output_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_products": len(products),
                    "generator_version": "1.0.0"
                },
                "products": [product.to_dict() for product in products]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Saved {len(products)} products to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving products to file: {e}")

    async def cleanup(self):
        """Clean up resources including shared agent"""
        try:
            # Clean up shared agent
            if self._shared_agent and self.project_client:
                try:
                    self.project_client.agents.delete_agent(self._shared_agent.id)
                    logger.info(f"Cleaned up shared agent: {self._shared_agent.id}")
                    self._shared_agent = None
                except Exception as e:
                    logger.debug(f"Failed to delete shared agent: {e}")
            
            logger.info("Agent Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate fashion product data with Azure AI Foundry grounding")
    parser.add_argument("--count", type=int, default=None, help="Number of products to generate (default: all products in CSV)")
    parser.add_argument("--csv-path", default="fashion_products.csv", help="Path to fashion CSV file containing product data")
    parser.add_argument("--output", default=None, help="Output JSON file path (default: output/fashion_products_YYYYMMDD_HHMMSS.json)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Determine how many products to process
    if args.count is None:
        # Load CSV to get total count
        try:
            import pandas as pd
            df = pd.read_csv(args.csv_path)
            total_products = len(df)
            args.count = total_products
            logger.info(f"No --count specified, processing all {total_products} products from CSV")
        except Exception as e:
            logger.error(f"Failed to read CSV file {args.csv_path}: {e}")
            logger.info("Defaulting to 5 products")
            args.count = 5
    else:
        logger.info(f"Processing {args.count} products as specified")

    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("output", exist_ok=True)
        args.output = f"output/fashion_products_{timestamp}.json"
    
    try:
        generator = FashionDataGenerator()
        
        # Generate products
        products = await generator.generate_fashion_product_data(args.count, args.csv_path)
        
        if products:
            # Save to file
            await generator.save_products_to_file(products, args.output)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"FASHION PRODUCT DATA GENERATION SUMMARY")
            print(f"{'='*60}")
            print(f"Output file: {args.output}")
            print(f" Total products: {len(products)}")
            print(f" Brands covered: {len(set(p.brandName for p in products))}")
            print(f"{'='*60}")
        else:
            logger.warning("No products were generated")

        # Cleanup
        await generator.cleanup()
        logger.info(f"Fashion product data generation completed successfully!")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
