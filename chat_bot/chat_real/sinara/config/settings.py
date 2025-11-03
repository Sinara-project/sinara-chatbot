from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Configurações globais da aplicação"""
    
    # API
    API_TITLE: str = "Chatbot Sinara"
    API_DESCRIPTION: str = "API do chatbot para suporte em ETAs"
    API_VERSION: str = "1.0.0"
    
    # IA (Gemini principal)
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-pro"
    MODEL_TEMP: float = 0.1

    # IA (modelos adicionais do Gemini)
    GEMINI_MODEL_GUARDRAIL: Optional[str] = None
    GEMINI_MODEL_JUDGE: Optional[str] = None
    GEMINI_MODEL_ASSISTENTE: Optional[str] = None
    GEMINI_MODEL_TECNICO: Optional[str] = None
    GEMINI_MODEL_ORG: Optional[str] = None
    GEMINI_CHAT_MODEL: Optional[str] = None

    # Banco de Dados MongoDB
    MONGO_URI: Optional[str] = None
    MONGO_DB: Optional[str] = None
    
    # RAG (busca de contexto)
    RAG_TOP_K: int = 5

    
    # Configuração geral
    
    class Config:
        env_file = ".env"

settings = Settings()
