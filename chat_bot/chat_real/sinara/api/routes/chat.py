from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging

from ...core.pipeline import run_pipeline
from ...services.rag_service import retrieve_similar_context
from ...agents.router_agent import run_router_agent
from ...api.models.requests import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatResponse(BaseModel):
    """Modelo de resposta do chat"""
    ok: bool
    agent: str
    session_id: Optional[str] = None
    contexts: List[str] = []
    answer: str

@router.get("/health", tags=["health"])
async def health_check():
    """Verifica status da API"""
    return {"status": "healthy"}

@router.get("/chat", response_model=ChatResponse, tags=["chat"])
async def chat_get(
    query: str,
    session_id: Optional[str] = None,
    agent: str = "auto"
):
    """
    Endpoint GET para consultas via URL
    
    Parâmetros:
        query: Pergunta do usuário
        session_id: ID da sessão (opcional)
        agent: Tipo de agente (padrão: auto)
    """
    request = ChatRequest(
        query=query,
        session_id=session_id,
        agent=agent
    )
    return await chat_endpoint(request)

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint principal para processar consultas"""
    try:
        logger.info(f"Consulta recebida: {request.query}")
        
        # Recupera contextos
        contexts = retrieve_similar_context(request.query)
        logger.debug(f"Contextos encontrados: {len(contexts) if contexts else 0}")
        
        # Define agente
        resolved_agent = request.agent
        if request.agent == "auto":
            resolved_agent, reason = run_router_agent(request.query, request.session_id)
            logger.info(f"Agente escolhido: {resolved_agent} ({reason})")

        # Processa resposta
        answer = run_pipeline(
            query=request.query,
            session_id=request.session_id,
            agent=resolved_agent,
            contexts=contexts
        )
        
        logger.debug(f"Resposta bruta do pipeline: tipo={type(answer)}, valor={str(answer)[:200]}")
        
        # Tratamento robusto da resposta
        if answer is None:
            answer = "Desculpe, não foi possível gerar uma resposta."
        elif isinstance(answer, tuple):
            answer = str(answer[0] if answer and answer[0] else "Resposta não disponível")
        elif isinstance(answer, (list, dict)):
            answer = str(answer)
        else:
            answer = str(answer)
            
        # Remove quebras de linha extras e espaços
        answer = answer.strip()
        
        response = ChatResponse(
            ok=True,
            agent=resolved_agent,
            session_id=request.session_id,
            contexts=contexts if isinstance(contexts, list) else [],
            answer=answer
        )
        
        logger.info(f"Resposta final gerada: {answer[:200]}")
        return response

    except Exception as e:
        logger.exception("Erro no processamento")
        return ChatResponse(
            ok=False,
            agent=resolved_agent if 'resolved_agent' in locals() else "error",
            session_id=request.session_id,
            contexts=[],
            answer=f"Erro ao processar sua pergunta: {str(e)}"
        )
