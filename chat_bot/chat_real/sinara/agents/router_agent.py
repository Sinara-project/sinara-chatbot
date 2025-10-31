import os
import logging
from typing import Optional, Tuple, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from ..services.rag_service import retrieve_similar_context_with_scores


load_dotenv(override=True)

logger = logging.getLogger(__name__)

# System-related keywords that should route to FAQ/organizational
SYSTEM_KEYWORDS = {
    'página', 'perfil', 'login', 'acesso', 'dashboard', 'formulário', 'formulario',
    'notificação', 'alerta', 'ponto', 'registro', 'cadastro', 'permissão', 'permissao',
    'usuário', 'usuario', 'aplicativo', 'mobile', 'web', 'plataforma', 'sinara', 'sistema'
}


class RouterDecision(BaseModel):
    route: str = Field(description="'assistente' | 'tecnico' | 'organizacional'")
    reason: Optional[str] = Field(default=None, description="Motivo resumido da escolha")


def _get_model(model_name: Optional[str] = None):
    m = model_name or os.getenv("GEMINI_MODEL_ROUTER") or os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash-latest")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("API key ausente")
    return ChatGoogleGenerativeAI(model=m, google_api_key=api_key).with_structured_output(RouterDecision)


_SYSTEM = (
    "system",
    (
        "Você é o agente ROTEADOR. Escolha exatamente UM destino entre: 'assistente', 'tecnico', 'organizacional'.\n\n"
        "Padrões de decisão:\n"
        "- 'assistente': dúvidas gerais, como usar o app, funcionalidades e FAQ de uso.\n"
        "- 'tecnico'   : questões técnicas de código/infra/API/erros/stack.\n"
        "- 'organizacional': políticas, processos, regras internas e institucionais.\n\n"
        "Retorne JSON com as chaves: route, reason. route ∈ {assistente|tecnico|organizacional}.\n"
    ),
)

_EXAMPLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template("{human}"),
        AIMessagePromptTemplate.from_template("{ai}"),
    ]
)

_FEWSHOTS = FewShotChatMessagePromptTemplate(
    examples=[
        {"human": "Como uso o formulário para registrar uma ocorrência?", "ai": '{"route":"assistente","reason":"Dúvida de uso"}'},
        {"human": "Qual é a string de conexão do Mongo e como criar o índice?", "ai": '{"route":"tecnico","reason":"Banco/infra"}'},
        {"human": "Quais são as regras para solicitar férias?", "ai": '{"route":"organizacional","reason":"Política institucional"}'},
        {"human": "Como bater ponto no sistema?", "ai": '{"route":"assistente","reason":"FAQ de uso (bater ponto)"}'},
        {"human": "Erro 500 no endpoint /chat", "ai": '{"route":"tecnico","reason":"Erro de API"}'},
    ],
    example_prompt=_EXAMPLE_PROMPT,
)

_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        _SYSTEM,
        _FEWSHOTS,
        ("human", "Escolha a rota para a mensagem a seguir.\\nMensagem: {query}"),
    ]
)


def _is_system_query(query: str) -> bool:
    """Check if query is about system features rather than technical water treatment"""
    query_words = set(query.lower().split())
    return bool(query_words & SYSTEM_KEYWORDS)


def run_router_agent(query: str, session_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Decide qual agente deve responder a 'query'.

    Estratégia:
      1) Se similaridade no contexto.json for alta, roteia para 'assistente'.
      2) Classificação LLM estruturada entre assistente/tecnico/organizacional.
      3) Heurística simples como fallback.
    """
    qtext = (query or "").strip()
    if not qtext:
        return "assistente", "Query vazia"

    # Heurística prioritária: técnico/organizacional
    q = qtext.lower()
    tecnico_kw_early = [
        "stack", "api", "endpoint", "erro", "traceback", "docker", "kubernetes",
        "deploy", "embedding", "pymongo", "langchain", "código", "bug", "faiss",
        "ph", "turbidez", "ntu", "alcalinidade", "cloração", "dosagem", "dosar", "coagulação",
        "floculação", "decantação", "filtração", "bomba dosadora", "eta", "hipoclorito",
        "sulfato de alumínio", "polímero", "jar test", "coagulante"
    ]
    organizacional_kw_early = ["política", "processo", "regra", "procedimento", "aluno", "matrícula", "férias", "documentação", "institucional"]

    if any(k in q for k in tecnico_kw_early):
        return "tecnico", "Heurística: termos técnicos (early)"
    if any(k in q for k in organizacional_kw_early):
        return "organizacional", "Heurística: termos organizacionais (early)"

    # 1) Sinal de FAQ pelo contexto.json
    try:
        pairs = retrieve_similar_context_with_scores(qtext, top_k=3)
        top_score = pairs[0][0] if pairs else 0.0
        if top_score >= 0.65:
            return "faq", f"FAQ match score={top_score:.2f}"
    except Exception:
        pass

    # 2) Classificação via LLM
    try:
        model = _get_model()
        chain = _ROUTER_PROMPT | model
        out: RouterDecision = chain.invoke({"query": qtext})
        route = getattr(out, "route", None) or "assistente"
        route = route.strip().lower()
        if route not in ("assistente", "tecnico", "organizacional"):
            route = "assistente"
        reason = getattr(out, "reason", None)
        return route, reason
    except Exception:
        pass

    # 3) Heurística simples
    q = qtext.lower()
    tecnico_kw = [
        "stack", "api", "endpoint", "erro", "traceback", "docker", "kubernetes",
        "deploy", "embedding", "pymongo", "langchain", "código", "bug", "faiss",
        # termos de ETA/operacional técnico
        "ph", "turbidez", "ntu", "alcalinidade", "cloração", "dosagem", "dosar", "coagulação",
        "floculação", "decantação", "filtração", "bomba dosadora", "eta", "hipoclorito",
        "sulfato de alumínio", "polímero", "jar test", "coagulante"
    ]
    organizacional_kw = ["política", "processo", "regra", "procedimento", "aluno", "matrícula", "férias", "documentação", "institucional"]

    if any(k in q for k in tecnico_kw):
        return "tecnico", "Heurística: termos técnicos"
    if any(k in q for k in organizacional_kw):
        return "organizacional", "Heurística: termos organizacionais"
    
    # Check for system-related queries
    if _is_system_query(qtext):
        logger.info("Query matched system keywords, routing to FAQ")
        return "faq", "Query relates to system features"

    return "assistente", "Heurística: padrão"

