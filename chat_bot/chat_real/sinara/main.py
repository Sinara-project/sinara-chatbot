import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config.settings import Settings
from .utils.logging_config import setup_logging
from .api.routes.chat import router as chat_router

# Configuração inicial
settings = Settings()  # cria instância de configurações
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Rotas
app.include_router(chat_router, prefix="/api")

#adicionando endpoint de health check
@app.get("/health")
async def health_check():
    """Verifica status da API"""
    return {"status": "healthy"}
