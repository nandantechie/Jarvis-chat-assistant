import os
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the application"""
    
    # API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Alternative name
    
    # Use whichever is available
    if not GOOGLE_API_KEY and GEMINI_API_KEY:
        GOOGLE_API_KEY = GEMINI_API_KEY
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-exp")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # LLM Parameters
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Database Configuration
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "scrapmate_rag")
    
    # Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    
    # Security Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "scrapmate-secret-key-2024")
    
    # File Upload Configuration
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "200")) * 1024 * 1024  # 200MB
    ALLOWED_EXTENSIONS = {'pdf'}
    
    @classmethod
    def is_valid(cls) -> bool:
        """Check if configuration is valid"""
        return bool(cls.GOOGLE_API_KEY)
    
    @classmethod
    def get_missing_configs(cls) -> List[str]:
        """Get list of missing required configurations"""
        missing = []
        if not cls.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        return missing
    
    @classmethod
    def validate_file(cls, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS