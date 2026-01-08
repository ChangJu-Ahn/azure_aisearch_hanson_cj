#!/usr/bin/env python3
"""
Fashion Content Analyzer using Azure AI Content Understanding
Azure AI Content Understanding을 사용하여 패션 상품 이미지를 상세 분석하고
기존 Azure AI Search 인덱스에 AI 분석 필드들을 추가합니다.

주요 기능:
- fashion_products.csv의 image_link에서 이미지 상세 분석
- Azure AI Content Understanding API를 통한 종합적 패션 아이템 분석
- 기존 fashion-sample 인덱스에 AI 분석 필드들 추가
- 배치 처리 및 에러 핸들링

사용법:
    python fashion_content_analyzer.py --csv fashion_products.csv
    python fashion_content_analyzer.py --csv fashion_products.csv --batch_size 5

필수 환경 변수 (.env 파일):
    AZURE_CONTENT_UNDERSTANDING_ENDPOINT=https://your-service.cognitiveservices.azure.com/
    AZURE_CONTENT_UNDERSTANDING_KEY=your-content-understanding-key
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
import json
import requests
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
content_understanding_endpoint = os.getenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT")
content_understanding_key = os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY")
content_project_name = os.getenv("FASHION_CONTENT_PROJECT_NAME", "fashion-image-analyzer")
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
class ContentAnalysisResult:
    """Content Understanding analysis result"""
    material: Optional[str] = None
    style: Optional[str] = None
    occasion: Optional[List[str]] = None
    clothing_type: Optional[str] = None
    primary_color: Optional[str] = None
    pattern: Optional[List[str]] = None
    fit: Optional[str] = None
    sleeve_length: Optional[str] = None
    neckline: Optional[str] = None
    length: Optional[str] = None
    season: Optional[List[str]] = None
    formality: Optional[str] = None
    target_gender: Optional[str] = None
    target_age: Optional[str] = None
    details: Optional[List[str]] = None
    brand_logo: Optional[str] = None
    text_on_image: Optional[str] = None
    caption: Optional[str] = None
    quality_flags: Optional[List[str]] = None


class FashionContentAnalyzer:
    """Analyze fashion product images using Azure AI Content Understanding"""
    
    # API configuration constants
    # 25.09.18 기준으로 2025-05-01-preview 버전만!! 여기 있는 기능들을 활용하실 수 있습니다. 
    API_VERSION = "2025-05-01-preview"
    
    def __init__(self):
        # Validate environment variables
        if not all([content_understanding_endpoint, content_understanding_key, search_endpoint, search_admin_key]):
            raise ValueError("Missing required environment variables. Please check your .env file.")
        
        # Content Understanding configuration
        self.content_understanding_endpoint = content_understanding_endpoint
        self.content_understanding_key = content_understanding_key
        self.fashion_analyzer_id = content_project_name or "fashion-image-analyzer"  # Use consistent naming
        self.api_version = None  # Will be set when first API call succeeds
        
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
        
        # TODO: Initialize Content Understanding client
        # This will be implemented once we have the actual Content Understanding SDK
        logger.info(f"Initialized Fashion Content Analyzer for index: {fashion_index_name}")
        logger.info(f"Content Understanding project: {content_project_name}")
    
    def get_fashion_analysis_schema(self) -> Dict[str, Any]:
        """Get the fashion image analysis schema for Content Understanding"""
        return {
            "name": "fashion-image-analyzer",
            "version": "0.1.0",
            "modality": "image",
            "language": "ko-KR",
            "schema": {
                "fields": [
                    {
                        "name": "material",
                        "type": "classification",
                        "description": "가장 중심/전경의 메인 의류의 주된 소재를 선택. 사진의 질감/광택/짜임을 근거로 판단. 로고/텍스트/배경은 제외.",
                        "classes": [
                            {"name": "알수없음", "description": "판단 곤란"},
                            {"name": "면", "description": "면, 매트하고 부드러운 섬유감"},
                            {"name": "데님", "description": "청지, 인디고/워싱/스티치"},
                            {"name": "울/니트", "description": "울/니트 계열, 보송한 조직"},
                            {"name": "가죽", "description": "가죽/레더, 광택/주름결"},
                            {"name": "리넨", "description": "마, 성근 짜임과 자연 주름"},
                            {"name": "실크/새틴", "description": "강한 광택과 유연한 드레이프"},
                            {"name": "합성섬유", "description": "폴리/나일론 등 합성섬유"},
                            {"name": "혼방", "description": "혼방으로 추정"}
                        ]
                    },
                    {
                        "name": "style",
                        "type": "classification",
                        "description": "가장 중심/전경의 메인 패션 아이템의 전반적 스타일 톤",
                        "classes": [
                            {"name": "캐주얼"}, {"name": "비즈니스"}, {"name": "스트리트"},
                            {"name": "페미닌"}, {"name": "포멀"}, {"name": "아웃도어"}
                        ]
                    },
                    {
                        "name": "occasion",
                        "type": "classification",
                        "multi_select": True,
                        "description": "이미지 전경에 배치된 메인 패션 아이템의 무드를 고려한 착용 상황/룩 태그(복수 선택 가능).",
                        "classes": [
                            {"name": "알수없음"},{"name": "비즈니스"}, {"name": "캐주얼"}, {"name": "하객"},
                            {"name": "모임"}, {"name": "데이트"}, {"name": "여행"}
                        ]
                    },
                    {
                        "name": "clothing_type",
                        "type": "classification",
                        "description": "가장 중심/전경의 메인 패션 아이템의 아이템 타입(상세 카테고리).",
                        "classes": [
                            {"name": "셔츠/블라우스"}, {"name": "니트/스웨터"}, {"name": "티셔츠"},
                            {"name": "원피스"}, {"name": "재킷/블레이저"}, {"name": "조끼"}, {"name": "코트"},
                            {"name": "스커트"}, {"name": "바지"}, {"name": "슬랙스"}, {"name": "청바지"},
                            {"name": "가디건"}, {"name": "후드/맨투맨"}, {"name": "셋업"},
                            {"name": "기타"}
                        ]
                    },
                    {
                        "name": "primary_color",
                        "type": "classification",
                        "description": "배경색은 제외하고 전경의 메인 아이템 기준으로 판단, 대표 색상 한 가지. 인간 모델의 피부색 및 배경 제외.",
                        "classes": [
                            {"name": "검정"}, {"name": "흰색"}, {"name": "회색"}, {"name": "네이비"},
                            {"name": "파랑"}, {"name": "초록"}, {"name": "빨강"}, {"name": "분홍"},
                            {"name": "베이지"}, {"name": "갈색"}, {"name": "노랑"}, {"name": "보라"},
                            {"name": "기타"}
                        ]
                    },
                    {
                        "name": "pattern",
                        "type": "classification",
                        "multi_select": True,
                        "description": "가장 중심/전경의 메인 패션 아이템의 무늬/패턴 태그.",
                        "classes": [
                            {"name": "무지"}, {"name": "스트라이프"}, {"name": "체크/플래드"},
                            {"name": "도트"}, {"name": "플로럴"}, {"name": "그래픽/로고"},
                            {"name": "애니멀"}, {"name": "케이블니트"}, {"name": "기타"}
                        ]
                    },
                    {
                        "name": "fit",
                        "type": "classification",
                        "description": "가장 중심/전경의 메인 패션 아이템인 의상의 실루엣/핏.",
                        "classes": [
                            {"name": "슬림"}, {"name": "레귤러"}, {"name": "릴렉스드"}, {"name": "오버사이즈"},
                            {"name": "테일러드"}, {"name": "알수없음"}
                        ]
                    },
                    {
                        "name": "sleeve_length",
                        "type": "classification",
                        "description": "가장 중심/전경의 메인 아이템이 상의인 경우 해당 상의의 소매 길이.",
                        "classes": [
                            {"name": "민소매"}, {"name": "반팔"}, {"name": "7부소매"}, {"name": "긴팔"},
                            {"name": "5부소매"}, {"name": "알수없음"}
                        ]
                    },
                    {
                        "name": "neckline",
                        "type": "classification",
                        "description": "가장 중심/전경의 메인 아이템이 상의인 경우 해당 상의의 넥라인/카라 형태.",
                        "classes": [
                            {"name": "라운드넥"}, {"name": "브이넥"}, {"name": "카라"}, {"name": "스퀘어넥"},
                            {"name": "터틀/목폴라"}, {"name": "오프숄더"}, {"name": "기타"}
                        ]
                    },
                    {
                        "name": "length",
                        "type": "classification",
                        "description": "가장 중심/전경의 메인 아이템이 하의인 경우 총장/스커트/드레스 길이(상대 구분).",
                        "classes": [
                            {"name": "크롭"}, {"name": "레귤러"}, {"name": "롱"},
                            {"name": "미니"}, {"name": "미디"}, {"name": "맥시"}, {"name": "알수없음"}
                        ]
                    },
                    {
                        "name": "season",
                        "type": "classification",
                        "multi_select": True,
                        "description": "가장 중심/전경의 메인 패션 아이템을 착용하기 적합한 계절(복수 가능).",
                        "classes": [
                            {"name": "봄"}, {"name": "여름"}, {"name": "가을"},
                            {"name": "겨울"}, {"name": "사계절"}
                        ]
                    },
                    {
                        "name": "formality",
                        "type": "classification",
                        "description": "가장 중심/전경의 메인 패션 아이템에 해당하는 포멀도 척도.",
                        "classes": [
                            {"name": "캐주얼"}, {"name": "스마트캐주얼"}, {"name": "세미포멀"}, {"name": "포멀"}
                        ]
                    },
                    {
                        "name": "target_gender",
                        "type": "classification",
                        "description": "가장 중심/전경의 메인 패션 아이템의 타깃 성별(이미지/스타일 맥락 기준 추정).",
                        "classes": [
                            {"name": "여성"}, {"name": "남성"}, {"name": "유니섹스"}, {"name": "알수없음"}
                        ]
                    },
                    {
                        "name": "target_age",
                        "type": "classification",
                        "description": "가장 중심/전경의 메인 패션 아이템의 적합 연령대(추정).",
                        "classes": [
                            {"name": "알수없음"}, {"name": "10대"}, {"name": "20대"}, {"name": "30대이상"}, {"name": "아동"}
                        ]
                    },
                    {
                        "name": "details",
                        "type": "classification",
                        "multi_select": True,
                        "description": "가장 중심/전경의 메인 패션 아이템의 디테일 요소 태그.",
                        "classes": [
                            {"name": "디테일없음"}, {"name": "단추"}, {"name": "지퍼"}, {"name": "벨트"}, {"name": "주머니"},
                            {"name": "러플"}, {"name": "플리츠"}, {"name": "레이스"}, {"name": "자수"},
                            {"name": "로고프린트"}, {"name": "스팽글"}, {"name": "기타"}
                        ]
                    },
                    {
                        "name": "brand_logo",
                        "type": "classification",
                        "description": "가장 중심/전경의 메인 패션 아이템의 브랜드 로고가 식별 가능한 경우 브랜드명(미식별 시 Unknown).",
                        "classes": [
                            {"name": "감지됨"}, {"name": "알수없음"}
                        ],
                        "notes": "필요 시 Vision의 Brand Detection 결과와 매핑"
                    },
                    {
                        "name": "text_on_image",
                        "type": "extract",
                        "description": "가장 중심/전경의 메인 패션 아이템의 텍스트 OCR 결과(라벨/태그/그래픽 텍스트)."
                    },
                    {
                        "name": "caption",
                        "type": "generate",
                        "description": "가장 중심/전경의 메인 패션 아이템을 표현하는 사람이 읽기 좋은 한 줄 설명(색상·실루엣·아이템 포함, 15~25자)."
                    },
                    {
                        "name": "quality_flags",
                        "type": "classification",
                        "multi_select": True,
                        "description": "운영 품질 점검 플래그.",
                        "classes": [
                            {"name": "저해상도"}, {"name": "가려짐"}, {"name": "여러아이템"},
                            {"name": "모델착용"}, {"name": "플랫레이"}, {"name": "마네킹"}
                        ]
                    }
                ],
                "output": {
                    "include_confidence": True,
                    "include_evidence": True,
                    "format": "json"
                },
                "guidance": {
                    "general": [
                        "모든 경우에 이미지 전경의 메인에 해당하는 패션 아이템 한 가지 중심으로 판단",
                        "상품이 복수 개 보이면 가장 중심/전경의 메인 아이템을 기준으로 판단",
                        "배경색 및 인간의 피부색은 primary_color 판단에 사용하지 않음",
                        "브랜드/매장 워터마크,문구는 속성 판단에 사용하지 않음",
                    ],
                    "tie_breakers": [
                        "드레스 vs 블라우스+스커트 혼동 시 드레스 우선",
                        "니트 조직이 뚜렷하면 니트/스웨터 우선"
                    ]
                }
            }
        }
    
    def _convert_schema_to_api_format(self) -> Dict[str, Any]:
        """Convert detailed fashion schema to Content Understanding API fieldSchema format"""
        detailed_schema = self.get_fashion_analysis_schema()
        api_fields = {}
        
        for field in detailed_schema["schema"]["fields"]:
            field_name = field["name"]
            field_type = field["type"]
            
            if field_type == "classification":
                # Handle classes_ref (reference to another field's classes)
                if "classes_ref" in field:
                    # Find the referenced field to get its classes
                    ref_field_name = field["classes_ref"]
                    ref_field = next((f for f in detailed_schema["schema"]["fields"] if f["name"] == ref_field_name), None)
                    if ref_field and "classes" in ref_field:
                        enum_values = [cls["name"] for cls in ref_field["classes"]]
                    else:
                        # Fallback: use primary_color classes if reference not found
                        enum_values = ["검정", "흰색", "주황", "회색", "네이비", "파랑", "초록", "빨강", "분홍", "베이지", "갈색", "노랑", "보라", "기타"]
                elif "classes" in field:
                    # Extract class names from classes list
                    enum_values = [cls["name"] for cls in field["classes"]]
                else:
                    # Skip fields without proper class definition
                    logger.warning(f"Skipping field {field_name}: no classes or classes_ref found")
                    continue
                
                if field.get("multi_select", False):
                    # Multi-select classification fields - require 'items' property for array type
                    api_fields[field_name] = {
                        "type": "array",
                        "method": "classify",
                        "items": {
                            "type": "string",
                            "enum": enum_values
                        }
                    }
                else:
                    # Single-select classification fields
                    api_fields[field_name] = {
                        "type": "string",
                        "method": "classify",
                        "enum": enum_values
                    }
            elif field_type == "extract":
                # Text extraction fields (OCR) - use simple string type
                api_fields[field_name] = {
                    "type": "string"
                }
            elif field_type == "generate":
                # Text generation fields (caption)
                api_fields[field_name] = {
                    "type": "string",
                    "method": "generate"
                }
            else:
                logger.warning(f"Skipping field {field_name}: unknown type {field_type}")
        
        try:
            logger.info(f"Generated API schema with {len(api_fields)} fields")
            logger.debug(f"API fields keys: {list(api_fields.keys())}")
            return api_fields
        except Exception as e:
            logger.error(f"Error in _convert_schema_to_api_format: {e}")
            logger.error(f"api_fields type: {type(api_fields)}")
            raise
    
    def create_content_understanding_project(self) -> bool:
        """Create or update Content Understanding analyzer with fashion analysis schema"""
        try:
            logger.info(f"Creating Content Understanding analyzer: {self.fashion_analyzer_id}")
            
            # Get the complete fashion analysis schema converted to API format
            api_fields = self._convert_schema_to_api_format()
            
            # Prepare analyzer payload according to Content Understanding API format
            analyzer_payload = {
                "description": "Fashion image analyzer for e-commerce fashion products with comprehensive 19-field analysis",
                "baseAnalyzerId": "prebuilt-imageAnalyzer",
                "config": {
                    "disableContentFiltering": False
                },
                "fieldSchema": {
                    "fields": api_fields
                }
            }

            # Create analyzer API call
            headers = {
                "Ocp-Apim-Subscription-Key": self.content_understanding_key,
                "Content-Type": "application/json"
            }
            
            create_url = f"{self.content_understanding_endpoint}/contentunderstanding/analyzers/{self.fashion_analyzer_id}"
            
            response = requests.put(
                create_url,
                headers=headers,
                json=analyzer_payload,
                params={"api-version": self.API_VERSION},
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                # Store successful API version for future calls
                self.api_version = self.API_VERSION
                logger.info(f"Analyzer '{self.fashion_analyzer_id}' created successfully")
                
                # Get operation location to check creation status
                operation_location = response.headers.get('Operation-Location')
                if operation_location:
                    self._wait_for_analyzer_creation(operation_location)
                
                return True
            else:
                logger.error(f"Failed to create analyzer: {response.status_code} - {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to create Content Understanding analyzer: {e}")
            return False
    
    def _wait_for_analyzer_creation(self, operation_location: str) -> bool:
        """Wait for analyzer creation to complete"""
        try:
            max_wait_time = 300  # 5 minutes
            poll_interval = 10   # Poll every 10 seconds
            elapsed_time = 0
            
            logger.info("Waiting for analyzer creation to complete...")
            
            while elapsed_time < max_wait_time:
                time.sleep(poll_interval)
                elapsed_time += poll_interval
                
                # Check operation status
                response = requests.get(
                    operation_location,
                    headers={"Ocp-Apim-Subscription-Key": self.content_understanding_key}
                )
                
                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data.get('status')
                    
                    if status == "Succeeded":
                        logger.info("Analyzer creation completed successfully")
                        return True
                    elif status == "Failed":
                        logger.error(f"Analyzer creation failed: {status_data.get('error', 'Unknown error')}")
                        return False
                    elif status in ["NotStarted", "Running"]:
                        logger.info(f"Analyzer creation in progress... ({elapsed_time}s)")
                        continue
                    else:
                        logger.warning(f"Unknown status: {status}")
                        continue
                else:
                    logger.warning(f"Failed to check operation status: {response.status_code}")
                    continue
            
            logger.warning("Analyzer creation timed out")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for analyzer creation: {e}")
            return False
    
    def delete_content_understanding_analyzer(self) -> bool:
        """Delete the Content Understanding analyzer"""
        try:
            logger.info(f"Deleting Content Understanding analyzer: {self.fashion_analyzer_id}")
            
            delete_url = f"{self.content_understanding_endpoint}/contentunderstanding/analyzers/{self.fashion_analyzer_id}"
            
            headers = {
                "Ocp-Apim-Subscription-Key": self.content_understanding_key
            }
            
            response = requests.delete(
                delete_url,
                headers=headers,
                params={"api-version": self.api_version or self.API_VERSION}
            )
            
            if response.status_code == 204:
                logger.info(f"✅ Analyzer '{self.fashion_analyzer_id}' deleted successfully")
                return True
            elif response.status_code == 404:
                logger.info(f"ℹ️ Analyzer '{self.fashion_analyzer_id}' not found (already deleted)")
                return True
            else:
                logger.error(f"❌ Failed to delete analyzer: {response.status_code} - {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to delete Content Understanding analyzer: {e}")
            return False
    
    def analyze_fashion_image(self, image_url: str) -> Optional[ContentAnalysisResult]:
        """Analyze fashion image using Azure AI Content Understanding"""
        try:
            logger.debug(f"Analyzing image: {image_url}")
            
            # Step 1: Submit analysis request
            analyze_url = f"{self.content_understanding_endpoint}/contentunderstanding/analyzers/{self.fashion_analyzer_id}:analyze"
            
            headers = {
                "Ocp-Apim-Subscription-Key": self.content_understanding_key,
                "Content-Type": "application/json"
            }
            
            # Request body with image URL
            request_body = {
                "url": image_url
            }
            
            # Submit analysis request
            response = requests.post(
                analyze_url,
                headers=headers,
                json=request_body,
                params={"api-version": self.api_version or self.API_VERSION}
            )
            
            if response.status_code != 202:
                logger.error(f"Failed to submit analysis request: {response.status_code} - {response.text}")
                return None
            
            # Get operation location from response headers
            operation_location = response.headers.get('Operation-Location')
            if not operation_location:
                logger.error("No Operation-Location header in response")
                return None
            
            # Extract operation ID from operation location
            operation_id = operation_location.split('/')[-1].split('?')[0]
            logger.debug(f"Analysis submitted, operation ID: {operation_id}")
            
            # Step 2: Poll for results
            max_wait_time = 60  # Maximum wait time in seconds
            poll_interval = 2   # Poll every 2 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                time.sleep(poll_interval)
                elapsed_time += poll_interval
                
                # Get analysis result
                result_url = f"{self.content_understanding_endpoint}/contentunderstanding/analyzerResults/{operation_id}"
                result_response = requests.get(
                    result_url,
                    headers={"Ocp-Apim-Subscription-Key": self.content_understanding_key},
                    params={"api-version": self.api_version or self.API_VERSION}
                )
                
                if result_response.status_code != 200:
                    logger.error(f"Failed to get analysis result: {result_response.status_code}")
                    continue
                
                result_data = result_response.json()
                status = result_data.get('status')
                
                if status == "Succeeded":
                    logger.debug(f"Analysis completed for: {image_url}")
                    return self._parse_content_understanding_result(result_data)
                elif status == "Failed":
                    logger.error(f"Analysis failed for image: {image_url}")
                    return None
                elif status in ["NotStarted", "Running"]:
                    logger.debug(f"Analysis in progress... Status: {status}")
                    continue
                else:
                    logger.warning(f"Unknown status: {status}")
                    continue
            
            logger.warning(f"Analysis timed out for image: {image_url}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze image {image_url}: {e}")
            return None
    
    def _parse_content_understanding_result(self, result_data: Dict[str, Any]) -> Optional[ContentAnalysisResult]:
        """Parse Content Understanding API response to ContentAnalysisResult"""
        try:
            if 'result' not in result_data:
                logger.error("No result field in response")
                return None
                
            result = result_data['result']
            
            # Extract contents and fields from the response
            contents = result.get('contents', [])
            if not contents:
                logger.warning("No contents in analysis result")
                return None
            
            # Get the first document content
            document = contents[0]
            fields = document.get('fields', {})
            
            # Parse fields according to our fashion schema with enum validation
            # Get valid enum values from schema for validation
            schema = self.get_fashion_analysis_schema()
            field_enums = {}
            for field_def in schema["schema"]["fields"]:
                if "classes" in field_def:
                    field_enums[field_def["name"]] = [cls["name"] for cls in field_def["classes"]]
            
            # Extract and validate single value fields
            material = self._validate_and_correct_enum_value(
                'material', 
                self._extract_field_value(fields, 'material', '알수없음'),
                field_enums.get('material', ['알수없음'])
            )
            style = self._validate_and_correct_enum_value(
                'style',
                self._extract_field_value(fields, 'style', '캐주얼'),
                field_enums.get('style', ['캐주얼'])
            )
            clothing_type = self._validate_and_correct_enum_value(
                'clothing_type',
                self._extract_field_value(fields, 'clothing_type', '알수없음'),
                field_enums.get('clothing_type', ['알수없음'])
            )
            primary_color = self._validate_and_correct_enum_value(
                'primary_color',
                self._extract_field_value(fields, 'primary_color', '알수없음'),
                field_enums.get('primary_color', ['알수없음'])
            )
            fit = self._validate_and_correct_enum_value(
                'fit',
                self._extract_field_value(fields, 'fit', '레귤러'),
                field_enums.get('fit', ['레귤러'])
            )
            sleeve_length = self._validate_and_correct_enum_value(
                'sleeve_length',
                self._extract_field_value(fields, 'sleeve_length', '알수없음'),
                field_enums.get('sleeve_length', ['알수없음'])
            )
            neckline = self._validate_and_correct_enum_value(
                'neckline',
                self._extract_field_value(fields, 'neckline', '알수없음'),
                field_enums.get('neckline', ['알수없음'])
            )
            length = self._validate_and_correct_enum_value(
                'length',
                self._extract_field_value(fields, 'length', '레귤러'),
                field_enums.get('length', ['레귤러'])
            )
            formality = self._validate_and_correct_enum_value(
                'formality',
                self._extract_field_value(fields, 'formality', '알수없음'),
                field_enums.get('formality', ['알수없음'])
            )
            target_gender = self._validate_and_correct_enum_value(
                'target_gender',
                self._extract_field_value(fields, 'target_gender', '유니섹스'),
                field_enums.get('target_gender', ['유니섹스'])
            )
            target_age = self._validate_and_correct_enum_value(
                'target_age',
                self._extract_field_value(fields, 'target_age', '알수없음'),
                field_enums.get('target_age', ['알수없음'])
            )
            
            # Array fields with improved parsing
            occasion_raw = self._extract_array_field(fields, 'occasion', ['알수없음'])
            pattern_raw = self._extract_array_field(fields, 'pattern', ['무지'])
            season_raw = self._extract_array_field(fields, 'season', ['사계절'])
            details_raw = self._extract_array_field(fields, 'details', ['디테일없음'])
            quality_flags_raw = self._extract_array_field(fields, 'quality_flags', [])
            
            # Validate array values against schema enums and remove duplicates
            occasion_validated = [self._validate_and_correct_enum_value('occasion', v, field_enums.get('occasion', ['알수없음'])) 
                                 for v in occasion_raw if v]
            occasion = list(dict.fromkeys(occasion_validated))  # Remove duplicates
            
            pattern_validated = [self._validate_and_correct_enum_value('pattern', v, field_enums.get('pattern', ['무지'])) 
                               for v in pattern_raw if v]
            pattern = list(dict.fromkeys(pattern_validated))  # Remove duplicates
            
            season_validated = [self._validate_and_correct_enum_value('season', v, field_enums.get('season', ['사계절'])) 
                               for v in season_raw if v]
            season = list(dict.fromkeys(season_validated))  # Remove duplicates
            
            details_validated = [self._validate_and_correct_enum_value('details', v, field_enums.get('details', ['디테일없음'])) 
                               for v in details_raw if v]
            details = list(dict.fromkeys(details_validated)) if details_validated else ['디테일없음']  # Remove duplicates or use default
            
            quality_flags_validated = [self._validate_and_correct_enum_value('quality_flags', v, field_enums.get('quality_flags', [])) 
                                     for v in quality_flags_raw if v]
            quality_flags = list(dict.fromkeys(quality_flags_validated))  # Remove duplicates
            
            # Additional fields
            brand_logo = self._extract_field_value(fields, 'brand_logo', '알수없음')
            text_on_image = self._extract_field_value(fields, 'text_on_image', '')
            
            # Generate meaningful caption from analysis results
            markdown_text = document.get('markdown', '')
            if markdown_text and not markdown_text.startswith('![image]'):
                # Use markdown if it's meaningful text (not just image placeholder)
                caption = markdown_text[:200]
            else:
                # Generate descriptive caption from AI analysis
                caption_parts = []
                if material != '알수없음':
                    caption_parts.append(f"{material} 소재")
                if style != '알수없음':
                    caption_parts.append(f"{style} 스타일")
                if clothing_type != '알수없음':
                    caption_parts.append(clothing_type)
                if primary_color != '알수없음':
                    caption_parts.append(f"{primary_color} 색상")
                
                caption = "의 ".join(caption_parts) if caption_parts else f"{style} {clothing_type}"
            
            return ContentAnalysisResult(
                material=material,
                style=style,
                occasion=occasion,
                clothing_type=clothing_type,
                primary_color=primary_color,
                pattern=pattern,
                fit=fit,
                sleeve_length=sleeve_length,
                neckline=neckline,
                length=length,
                season=season,
                formality=formality,
                target_gender=target_gender,
                target_age=target_age,
                details=details,
                brand_logo=brand_logo,
                text_on_image=text_on_image,
                caption=caption,
                quality_flags=quality_flags
            )
            
        except Exception as e:
            logger.error(f"Failed to parse Content Understanding result: {e}")
            return None
    
    def _extract_field_value(self, fields: Dict[str, Any], field_name: str, default: str) -> str:
        """Extract single string value from Content Understanding fields"""
        try:
            if field_name in fields:
                field_data = fields[field_name]
                if isinstance(field_data, dict) and 'valueString' in field_data:
                    return field_data['valueString']
                elif isinstance(field_data, str):
                    return field_data
            return default
        except Exception:
            return default
    
    def _extract_array_field(self, fields: Dict[str, Any], field_name: str, default: List[str]) -> List[str]:
        """Extract array values from Content Understanding fields"""
        try:
            if field_name in fields:
                field_data = fields[field_name]
                logger.debug(f"Extracting {field_name}: {type(field_data)} = {field_data}")
                
                # Handle different response formats from Content Understanding API
                if isinstance(field_data, list):
                    # Case 1: Direct list of values
                    result = []
                    for item in field_data:
                        if isinstance(item, dict) and 'valueString' in item:
                            # Format: [{'type': 'string', 'valueString': 'value1'}, ...]
                            result.append(item['valueString'])
                        elif isinstance(item, str):
                            # Format: ['value1', 'value2', ...]
                            result.append(item)
                        else:
                            logger.debug(f"Skipping unexpected item type in {field_name}: {type(item)} = {item}")
                    if result:
                        logger.debug(f"Successfully extracted {field_name} array: {result}")
                        return result
                
                elif isinstance(field_data, dict):
                    # Case 2: Single object with valueArray or valueString
                    if 'valueArray' in field_data:
                        logger.debug(f"Found valueArray in {field_name}: {field_data['valueArray']}")
                        return field_data['valueArray']
                    elif 'valueString' in field_data:
                        # Single value wrapped in object
                        logger.debug(f"Found single valueString in {field_name}: {field_data['valueString']}")
                        return [field_data['valueString']]
                
                elif isinstance(field_data, str):
                    # Case 3: Direct string value
                    logger.debug(f"Found direct string in {field_name}: {field_data}")
                    return [field_data]
                
                else:
                    logger.warning(f"Unexpected field data type for {field_name}: {type(field_data)} = {field_data}")
            
            logger.debug(f"Using default value for {field_name}: {default}")
            return default
        except Exception as e:
            logger.warning(f"Error extracting array field {field_name}: {e}")
            return default
    
    def _validate_and_correct_enum_value(self, field_name: str, value: str, valid_enums: List[str]) -> str:
        """Validate enum value and correct common mistakes"""
        if not value or not isinstance(value, str):
            return "알수없음"
        
        # Direct match
        if value in valid_enums:
            return value
        
        # Common corrections mapping
        corrections = {
            # Material corrections
            "리너": "리넨",
            "린넨": "리넨",
            "리넌": "리넨", 
            "코튼": "면",
            "cotton": "면",
            "denim": "데님",
            "leather": "가죽",
            "wool": "울/니트",
            "silk": "실크/새틴",
            
            # Color corrections  
            "베이비": "분홍",
            "baby": "분홍",
            "핑크": "분홍",
            "pink": "분홍",
            "블랙": "검정",
            "black": "검정",
            "화이트": "흰색",
            "white": "흰색",
            "그레이": "회색",
            "gray": "회색",
            "그린": "초록",
            "green": "초록",
            "녹색": "초록",  # 추가
            "레드": "빨강",
            "red": "빨강",
            "블루": "파랑",
            "blue": "파랑",
            "옐로우": "노랑",
            "yellow": "노랑",
            "퍼플": "보라",
            "purple": "보라",
            
            # Style corrections
            "casual": "캐주얼",
            "business": "비즈니스", 
            "street": "스트릿",
            "feminine": "페미닌",
            "formal": "포멀",
            "outdoor": "아웃도어"
        }
        
        # Check corrections mapping
        if value in corrections:
            corrected = corrections[value]
            if corrected in valid_enums:
                logger.info(f"Applied correction for {field_name}: '{value}' → '{corrected}'")
                return corrected
        
        # Fuzzy matching for close matches
        close_matches = difflib.get_close_matches(value, valid_enums, n=1, cutoff=0.6)
        if close_matches:
            corrected = close_matches[0]
            logger.info(f"Fuzzy matched {field_name}: '{value}' → '{corrected}'")
            return corrected
        
        # Special handling for style field - keep original value if not found
        if field_name == 'style':
            logger.info(f"Style field '{value}' not in enum, keeping original value")
            return value
        
        # Special handling for season field - keep original value if not found
        if field_name == 'season':
            logger.info(f"Season field '{value}' not in enum, keeping original value")
            return value
        
        # Special handling for occasion field - keep original value if not found
        if field_name == 'occasion':
            logger.info(f"Occasion field '{value}' not in enum, keeping original value")
            return value
        
        # Special handling for pattern field - keep original value if not found
        if field_name == 'pattern':
            logger.info(f"Pattern field '{value}' not in enum, keeping original value")
            return value
        
        # Special handling for details field - keep original value if not found
        if field_name == 'details':
            logger.info(f"Details field '{value}' not in enum, keeping original value")
            return value
        
        # Special handling for target_age field - reject gender values
        if field_name == 'target_age' and value in ['여성', '남성', '유니섹스']:
            logger.warning(f"Detected gender value '{value}' in target_age field, using default")
            return "알수없음"
        
        # For core classification fields, fallback to first valid enum or default
        logger.warning(f"Could not correct {field_name} value: '{value}'. Valid options: {valid_enums}. Using fallback: '{valid_enums[0] if valid_enums else '알수없음'}'")
        return valid_enums[0] if valid_enums else "알수없음"
    
    def format_analysis_result(self, result: ContentAnalysisResult) -> Dict[str, Any]:
        """Format analysis result for search index update"""
        if not result:
            return {}
        
        formatted = {}
        
        # Single value fields
        if result.material:
            formatted["ai_material"] = result.material
        if result.style:
            formatted["ai_style"] = result.style
        if result.clothing_type:
            formatted["ai_clothing_type"] = result.clothing_type
        if result.primary_color:
            formatted["ai_primary_color"] = result.primary_color
        if result.fit:
            formatted["ai_fit"] = result.fit
        if result.sleeve_length:
            formatted["ai_sleeve_length"] = result.sleeve_length
        if result.neckline:
            formatted["ai_neckline"] = result.neckline
        if result.length:
            formatted["ai_length"] = result.length
        if result.formality:
            formatted["ai_formality"] = result.formality
        if result.target_gender:
            formatted["ai_target_gender"] = result.target_gender
        if result.target_age:
            formatted["ai_target_age"] = result.target_age
        if result.brand_logo:
            formatted["ai_brand_logo"] = result.brand_logo
        if result.text_on_image:
            formatted["ai_text_on_image"] = result.text_on_image
        if result.caption:
            formatted["ai_caption"] = result.caption
        
        # Multi-value fields (arrays) - convert to comma-separated strings
        if result.occasion and isinstance(result.occasion, list):
            formatted["ai_occasion"] = ", ".join(str(item) for item in result.occasion if item)
        if result.pattern and isinstance(result.pattern, list):
            formatted["ai_pattern"] = ", ".join(str(item) for item in result.pattern if item)
        if result.season and isinstance(result.season, list):
            formatted["ai_season"] = ", ".join(str(item) for item in result.season if item)
        if result.details and isinstance(result.details, list):
            formatted["ai_details"] = ", ".join(str(item) for item in result.details if item)
        if result.quality_flags and isinstance(result.quality_flags, list):
            formatted["ai_quality_flags"] = ", ".join(str(item) for item in result.quality_flags if item)
        
        return formatted
    
    def add_content_fields_to_index(self) -> bool:
        """Add Content Understanding analysis fields to the index schema"""
        try:
            # Get current index schema
            current_index = self.index_client.get_index(fashion_index_name)
            current_fields = {field.name: field for field in current_index.fields}
            
            # Define new AI analysis fields
            new_fields = []
            
            # Single value string fields
            single_fields = [
                "ai_material", "ai_style", "ai_clothing_type", "ai_primary_color",
                "ai_fit", "ai_sleeve_length", "ai_neckline", "ai_length",
                "ai_formality", "ai_target_gender", "ai_target_age", "ai_brand_logo",
                "ai_text_on_image", "ai_caption"
            ]
            
            for field_name in single_fields:
                if field_name not in current_fields:
                    new_fields.append(SearchableField(
                        name=field_name,
                        type=SearchFieldDataType.String,
                        searchable=True,
                        filterable=True,
                        facetable=True
                    ))
            
            # Multi-value fields as String instead of Collection to avoid Azure Search issues
            multi_value_fields = [
                "ai_occasion", "ai_pattern", "ai_season",
                "ai_details", "ai_quality_flags"
            ]
            
            for field_name in multi_value_fields:
                if field_name not in current_fields:
                    new_fields.append(SearchableField(
                        name=field_name,
                        type=SearchFieldDataType.String,  # Use String instead of Collection
                        searchable=True,
                        filterable=True,
                        facetable=True
                    ))
            
            if new_fields:
                logger.info(f"Adding {len(new_fields)} new AI analysis fields to index schema...")
                
                # Update index schema
                updated_fields = list(current_index.fields) + new_fields
                
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
                logger.info("Successfully added AI analysis fields to index schema")
            else:
                logger.info("All AI analysis fields already exist in index schema")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add AI analysis fields to index: {e}")
            return False
    
    def process_csv_and_analyze(self, csv_path: str, batch_size: int = 10, count: int = None, max_workers: int = 10) -> List[Dict[str, Any]]:
        """Process CSV file and analyze fashion images with parallel processing"""
        try:
            df = pd.read_csv(csv_path)
            
            # Limit the number of rows if count is specified
            if count is not None:
                df = df.head(count)
                logger.info(f"Limiting to first {count} products from {csv_path}")
            
            logger.info(f"Loaded {len(df)} products from {csv_path}")
            logger.info(f"Using {max_workers} parallel workers for image analysis")
            
            analyzed_products = []
            
            # Process in batches to manage memory and API rate limits
            total_rows = len(df)
            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                batch_df = df.iloc[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}: rows {batch_start}-{batch_end-1}")
                
                # Prepare batch data for parallel processing
                batch_tasks = []
                for idx, row in batch_df.iterrows():
                    image_url = row.get('image_link', '')
                    product_id = row.get('id', '')
                    
                    if not image_url:
                        logger.warning(f"No image URL for product {product_id}")
                        continue
                    
                    batch_tasks.append((product_id, image_url))
                
                # Process batch in parallel
                batch_results = self._process_batch_parallel(batch_tasks, max_workers)
                analyzed_products.extend(batch_results)
                
                # Rate limiting between batches
                if batch_end < total_rows:
                    logger.debug(f"Batch {batch_start//batch_size + 1} completed. Waiting 2 seconds before next batch...")
                    time.sleep(2)
            
            logger.info(f"Successfully analyzed {len(analyzed_products)} products using parallel processing")
            return analyzed_products
            
        except Exception as e:
            logger.error(f"Failed to process CSV file: {e}")
            raise
    
    def _process_batch_parallel(self, batch_tasks: List[tuple], max_workers: int) -> List[Dict[str, Any]]:
        """Process a batch of images in parallel using ThreadPoolExecutor"""
        batch_results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_task = {
                executor.submit(self._analyze_single_image, product_id, image_url): (product_id, image_url)
                for product_id, image_url in batch_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                product_id, image_url = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        logger.debug(f"✅ Product {product_id}: {len(result)-1} AI fields analyzed")  # -1 for 'id' field
                    else:
                        logger.warning(f"❌ Product {product_id}: Analysis failed")
                        
                except Exception as e:
                    logger.error(f"❌ Product {product_id}: Exception during analysis - {e}")
        
        logger.info(f"Batch parallel processing completed: {len(batch_results)}/{len(batch_tasks)} successful")
        return batch_results
    
    def _analyze_single_image(self, product_id: str, image_url: str) -> Optional[Dict[str, Any]]:
        """Analyze a single image and return formatted result"""
        try:
            # Analyze image with Content Understanding
            analysis_result = self.analyze_fashion_image(image_url)
            
            if analysis_result:
                formatted_result = self.format_analysis_result(analysis_result)
                formatted_result["id"] = str(product_id)
                return formatted_result
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to analyze product {product_id} ({image_url}): {e}")
            return None
    
    def update_search_index(self, analyzed_data: List[Dict[str, Any]], batch_size: int = 10) -> bool:
        """Update search index with Content Understanding analysis results"""
        try:
            logger.info(f"Updating search index with {len(analyzed_data)} AI analysis records")
            
            # Debug: Log the raw analyzed data
            if analyzed_data:
                logger.debug(f"Raw analyzed data sample: {json.dumps(analyzed_data[0], indent=2, ensure_ascii=False)}")
            
            total_successful = 0
            total_failed = 0
            
            # Update in batches
            for i in tqdm(range(0, len(analyzed_data), batch_size), desc="Updating index"):
                batch = analyzed_data[i:i+batch_size]
                
                # Prepare documents for merge/upload
                documents_to_update = []
                for item in batch:
                    doc = {"@search.action": "merge"}  # Use merge to update existing documents
                    
                    # Clean up data types for Azure Search
                    for key, value in item.items():
                        if key == "id":
                            doc[key] = value
                        elif isinstance(value, list):
                            if value and any(v for v in value if v is not None and str(v).strip()):
                                clean_list = [str(v).strip() for v in value if v is not None and str(v).strip()]
                                if clean_list: 
                                    doc[key] = ", ".join(clean_list)
                            # If empty list or all None values, skip this field entirely
                        elif value is not None and str(value).strip():
                            # Ensure single values are strings and not empty
                            doc[key] = str(value).strip()
                    
                    documents_to_update.append(doc)
                
                # Debug: Log the processed document structure
                if documents_to_update:
                    logger.debug(f"Processed document sample: {json.dumps(documents_to_update[0], indent=2, ensure_ascii=False)}")
                
                try:
                    result = self.search_client.upload_documents(documents=documents_to_update)
                    
                    # Check for failures
                    if result:
                        successful_items = [item for item in result if item.succeeded]
                        failed_items = [item for item in result if not item.succeeded]
                        
                        total_successful += len(successful_items)
                        total_failed += len(failed_items)
                        
                        if failed_items:
                            logger.warning(f"Batch {i//batch_size + 1}: {len(successful_items)} succeeded, {len(failed_items)} failed")
                            for item in failed_items:
                                logger.error(f"Failed to update {item.key}: {item.error_message}")
                                # Debug: Log the problematic document
                                problematic_doc = next((doc for doc in documents_to_update if doc.get("id") == item.key), None)
                                if problematic_doc:
                                    logger.debug(f"Problematic document: {json.dumps(problematic_doc, indent=2, ensure_ascii=False)}")
                        else:
                            logger.debug(f"Batch {i//batch_size + 1}: All {len(successful_items)} documents updated successfully")
                
                except Exception as e:
                    logger.error(f"Batch {i//batch_size + 1} update failed: {e}")
                    # Debug: Log the documents that failed
                    logger.debug(f"Failed batch documents: {json.dumps(documents_to_update, indent=2, ensure_ascii=False)}")
                    total_failed += len(batch)
                    continue
            
            logger.info(f"Index update completed: {total_successful} successful, {total_failed} failed")
            return total_successful > 0
            
        except Exception as e:
            logger.error(f"Failed to update search index: {e}")
            return False
    
    def run_content_analysis(self, csv_path: str, batch_size: int = 10, cleanup_analyzer: bool = True, count: int = None, max_workers: int = 10) -> bool:
        """Complete workflow: analyze images and update index with parallel processing"""
        analyzer_created = False
        
        try:
            logger.info("Starting fashion content analysis workflow")
            logger.info(f"🔧 Configuration: batch_size={batch_size}, max_workers={max_workers}, cleanup={cleanup_analyzer}")
            
            # Step 1: Create Content Understanding analyzer
            logger.info("Step 1: Creating Content Understanding analyzer...")
            if self.create_content_understanding_project():
                analyzer_created = True
                logger.info("Content Understanding analyzer created successfully")
            else:
                logger.error("Failed to create Content Understanding analyzer")
                return False
            
            # Step 2: Add AI analysis fields to index
            logger.info("🔧 Step 2: Adding AI analysis fields to search index...")
            if not self.add_content_fields_to_index():
                logger.error("Failed to add AI fields to search index")
                return False
            logger.info("Search index schema updated successfully")
            
            # Step 3: Process CSV and analyze images in parallel
            logger.info("⚡ Step 3: Processing images with parallel Content Understanding analysis...")
            analyzed_data = self.process_csv_and_analyze(csv_path, batch_size, count, max_workers)
            
            if not analyzed_data:
                logger.error("No analysis data generated")
                return False
            logger.info(f"Successfully analyzed {len(analyzed_data)} fashion images with parallel processing")
            
            # Step 4: Update search index
            logger.info("Step 4: Updating search index with AI analysis results...")
            success = self.update_search_index(analyzed_data, batch_size)
            
            if success:
                logger.info("🎉 Fashion content analysis completed successfully")
                return True
            else:
                logger.error("❌ Failed to update search index")
                return False
                
        except Exception as e:
            logger.error(f"workflow failed: {e}")
            return False
        
        finally:
            # Cleanup: Delete analyzer if requested and it was created
            if cleanup_analyzer and analyzer_created:
                logger.info("Cleaning up: Deleting Content Understanding analyzer...")
                if self.delete_content_understanding_analyzer():
                    logger.info("Analyzer cleanup completed")
                else:
                    logger.warning("Failed to cleanup analyzer (may need manual deletion)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Analyze fashion product images using Azure AI Content Understanding"
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
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--keep-analyzer",
        action="store_true",
        help="Keep the Content Understanding analyzer after processing (default: delete after use)"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        help="Limit the number of images to process (for testing purposes)"
    )
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=10,
        help="Maximum number of parallel workers for image analysis (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.verbose)
    
    analyzer = None
    
    try:
        # Validate input file
        if not Path(args.csv).exists():
            logger.error(f"CSV file not found: {args.csv}")
            sys.exit(1)
        
        # Initialize analyzer
        analyzer = FashionContentAnalyzer()
        
        # Run content analysis
        logger.info("Starting fashion content analysis process")
        logger.info(f"📁 Input CSV: {args.csv}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"⚡ Max workers: {args.max_workers}")
        if args.count:
            logger.info(f"Processing limit: {args.count} images")
        logger.info(f"🧹 Cleanup analyzer: {not args.keep_analyzer}")
        
        success = analyzer.run_content_analysis(
            csv_path=args.csv, 
            batch_size=args.batch_size,
            cleanup_analyzer=not args.keep_analyzer,
            count=args.count,
            max_workers=args.max_workers
        )
        
        if success:
            print(f"\n{'='*70}")
            print(f"CONTENT UNDERSTANDING TASK COMPLETED SUCCESSFULLY 🎉")
            print(f"{'='*70}")
            print(f"Input file: {args.csv}")
            print(f"Target index: {fashion_index_name}")
            print(f"Analyzer: {analyzer.fashion_analyzer_id}")
            print(f"Status: SUCCESS")
            print(f"{'='*70}")
            logger.info("Content Analysis completed successfully")
        else:
            logger.error("Fashion content analysis failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        if analyzer:
            logger.info("Performing emergency cleanup...")
            try:
                analyzer.delete_content_understanding_analyzer()
                logger.info("Emergency cleanup completed (deleting analyzer)")
            except Exception as cleanup_error:
                logger.error(f"Emergency cleanup failed: {cleanup_error}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Content analysis process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
