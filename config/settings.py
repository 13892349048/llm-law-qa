"""
setting files 
"""

import os 
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import torch

class Settings(BaseSettings):
    """app settings"""

    # project root 
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = DATA_DIR / "models" 
    LOGS_DIR: Path = PROJECT_ROOT / "logs" 

    # data 
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    IMAGES_DIR: Path = DATA_DIR / "images"

    # CLIP model config
    CLIP_MODEL_NAME: str = "ViT-B-32"
    CLIP_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    #multimodal model config
    IMAGE_SIZE: int = 224
    MAX_TEXT_LENGTH: int = 100
    SUPPORT_IMAGE_FORMATS: list[str] = [".jpg", ".jpeg", ".png"]
    SUPPORT_DOC_FORMATS: list[str] = [".pdf", ".docx", ".doc"]


    #model config
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5" #中文法律嵌入模型
    LLM_MODEL: str = "qwen/2.5:7b"   #法律问答模型
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # text chunking config
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100
    MAX_CHUNK_SIZE: int = 1000
    MIN_CHUNK_SIZE: int = 100

    #ocr config
    OCR_LANGUAGE: str = "chi_sim+eng"
    OCT_CONFIDENCE: float = 0.6

    #DATABASE
    VECTOR_STORE_TYPE: str = "chroma"
    CHROMA_DB_PATH: str = str(DATA_DIR / "chroma_db") 
    EMBEDDING_DIM: int = 512

    #API KEYS
    OPENAI_API_KEY: Optional[str] = None
    PINNECONE_API_KEY: Optional[str] = None

    #retrieval config
    TOP_K_RETRIEVAL: int =10 
    TOP_K_RERANK: int = 5
    SIMILARITY_THRESHOLD: float = 0.7 

    #web config
    STREAMLIT_PORT: int = 8501

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

#global settings instance
settings = Settings()

#create directories if they don't exist
for dir in [settings.DATA_DIR, settings.MODELS_DIR, settings.LOGS_DIR, 
            settings.RAW_DATA_DIR, settings.PROCESSED_DATA_DIR, settings.IMAGES_DIR]:
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)

# easy test
#s = Settings()
#print(s.DATA_DIR)