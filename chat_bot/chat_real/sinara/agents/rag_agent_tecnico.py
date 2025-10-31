import os
import json
import logging
from typing import Tuple, List, Optional
from dotenv import load_dotenv
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

from ..services.memory_tecnico import get_memory
from ..services.rag_service import retrieve_similar_context, retrieve_similar_context_with_scores

# Configuração de logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(override=True)


def _get_chat_model(model_name: str) -> ChatGoogleGenerativeAI:
    """
    Inicializa o modelo de chat com configurações específicas.
    Args:
        model_name: Nome do modelo Gemini a ser usado
    Returns:
        Instância configurada do ChatGoogleGenerativeAI
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Chave de API não encontrada nas variáveis de ambiente")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3,
    )


# Carrega o prompt do sistema
system_prompt_path = os.path.join(
    os.path.dirname(__file__), "../prompts/rag/tecnico/system_prompt_tecnico.txt"
)
with open(system_prompt_path, "r", encoding="utf-8") as f:
    system_text = f.read()
system_prompt = ("system", system_text)

# Carrega exemplos para few-shot learning
fewshots_path = os.path.join(
    os.path.dirname(__file__), "../prompts/rag/tecnico/fewshot.json"
)
with open(fewshots_path, "r", encoding="utf-8") as f:
    shots = json.load(f)

# Configura o template para exemplos
example_prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template("{human}"),
        AIMessagePromptTemplate.from_template("{ai}"),
    ]
)

# Configura o few-shot learning
fewshots = FewShotChatMessagePromptTemplate(
    examples=shots,
    example_prompt=example_prompt,
)

# Template completo do prompt RAG
rag_prompt = ChatPromptTemplate.from_messages(
    [
        system_prompt,
        fewshots,
        MessagesPlaceholder("memory"),
        ("human", "Contexto:\n{context}\n\nPergunta:\n{query}"),
    ]
)


def run_rag_agent_tecnico(query: str, session_id: str) -> Tuple[str, str]:
    """
    Executa o agente técnico RAG para responder consultas sobre tratamento de água.
    Args:
        query: Pergunta do usuário
        session_id: Identificador da sessão para histórico
    Returns:
        Tupla (resposta, contexto_usado)
    """
    try:
        # Recupera contexto relevante
        ctx = retrieve_similar_context(query)
        context = "\n".join(ctx) if isinstance(ctx, list) else str(ctx or "")
        logger.debug(f"Contexto recuperado: {context[:200]}...")
    except Exception as e:
        logger.exception("Erro na recuperação de contexto")
        return (
            "Desculpe, houve um problema ao processar sua pergunta. Tente novamente mais tarde.",
            "",
        )

    try:
        # Configura modelo e gera resposta
        memory = get_memory(session_id)
        model_name = os.getenv("GEMINI_MODEL_TECNICO") or os.getenv(
            "GEMINI_CHAT_MODEL", "gemini-pro"
        )
        model = _get_chat_model(model_name)
        chain = rag_prompt | model

        # Invoca o modelo
        output = chain.invoke(
            {
                "context": context,
                "query": query,
                "memory": getattr(memory, "messages", []),
            }
        )

        content = getattr(output, "content", None) or str(output)
        logger.info(f"Resposta gerada (primeiros 200 chars): {content[:200]}")
        return content, context

    except Exception as e:
        logger.exception("Erro na geração da resposta")
        # Fallback para contextos similares se o modelo falhar
        pairs = retrieve_similar_context_with_scores(query, top_k=3)
        if pairs:
            _score, text = pairs[0]
            return (str(text).strip()[:1200], context)
        return ("Não encontrei informação técnica suficiente no contexto.", context)


