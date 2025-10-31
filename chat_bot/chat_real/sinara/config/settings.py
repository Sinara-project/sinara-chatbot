from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configurações globais da aplicação"""
    
    # API
    API_TITLE: str = "Chatbot Sinara"
    API_DESCRIPTION: str = "API do chatbot para suporte em ETAs"
    API_VERSION: str = "1.0.0"
    
    # IA
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-pro"
    MODEL_TEMP: float = 0.1
    
    # RAG
    RAG_TOP_K: int = 5
    
    class Config:
        env_file = ".env"

settings = Settings()