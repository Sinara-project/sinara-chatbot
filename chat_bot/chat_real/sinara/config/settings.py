from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configurações globais da aplicação"""
    
    # ===================================
    # API
    # ===================================
    API_TITLE: str = "Chatbot Sinara"
    API_DESCRIPTION: str = "API do chatbot para suporte em ETAs"
    API_VERSION: str = "1.0.0"
    
    # ===================================
    # IA (Gemini principal)
    # ===================================
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-pro"
    MODEL_TEMP: float = 0.1

    # ===================================
    # IA (modelos adicionais do Gemini)
    # ===================================
    GEMINI_MODEL_GUARDRAIL: str | None = None
    GEMINI_MODEL_JUDGE: str | None = None
    GEMINI_MODEL_ASSISTENTE: str | None = None
    GEMINI_MODEL_TECNICO: str | None = None
    GEMINI_MODEL_ORG: str | None = None
    GEMINI_CHAT_MODEL: str | None = None

    # ===================================
    # Banco de Dados MongoDB
    # ===================================
    MONGO_URI: str | None = None
    MONGO_DB: str | None = None
    
    # ===================================
    # RAG (busca de contexto)
    # ===================================
    RAG_TOP_K: int = 5

    # ===================================
    # Configuração geral
    # ===================================
    class Config:
        env_file = ".env"   # garante que as variáveis do arquivo .env sejam lidas

settings = Settings()
