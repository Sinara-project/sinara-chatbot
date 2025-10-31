import uuid
from typing import Iterable, List, Optional
import logging
import traceback

from ..services.memory_assistente import get_memory as get_memory_assistente
from ..services.memory_tecnico import get_memory as get_memory_tecnico
from ..agents.guardrail_agent import run_guardrail_agent
from ..agents.judge_agent import run_judge_agent
from ..agents.rag_agent_assistente import run_rag_agent_assistente
from ..agents.rag_agent_tecnico import run_rag_agent_tecnico
from ..agents.rag_agent_organizacional import run_rag_agent_organizacional
from ..agents.router_agent import run_router_agent
from ..agents.faq_agent import run_faq_agent

# Configuração de logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Palavras‑chave relacionadas ao sistema
SYSTEM_KEYWORDS = {
    'login', 'acesso', 'usuario', 'usuario', 'perfil', 'pagina', 'pagina',
    'dashboard', 'painel', 'formulario', 'formulario', 'notificacao', 'notificacao',
    'aplicativo', 'mobile', 'web', 'plataforma', 'sinara', 'sistema'
}


def _is_system_query(query: str) -> bool:
    query_words = set(query.lower().split())
    return bool(query_words & SYSTEM_KEYWORDS)


def _contexts_match_query(contexts: Optional[Iterable[str]], query: str) -> bool:
    if not contexts:
        return False
    q = query.lower().strip()
    tokens = set(t for t in q.split() if len(t) > 2)
    for ctx in contexts:
        if not isinstance(ctx, str):
            continue
        text = ctx.lower()
        if q and q in text:
            logger.info("frase encontrada no contexto")
            return True
        matches = sum(1 for t in tokens if t in text)
        if tokens and matches >= max(1, int(len(tokens) * 0.6)):
            logger.info(f"Não encontrado no contexto: {matches}/{len(tokens)}")
            return True
    return False


def _safe_call_agent(fn, *args, **kwargs):
    """Auxiliar para chamar funções de agentes com registro de exceções e fallback seguro."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("Agent call failed: %s", fn.__name__)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return ("Desculpe, ocorreu um erro ao processar sua pergunta.", "")


def run_pipeline(query: str, session_id: str | None = None, agent: str = "auto", contexts: list | None = None) -> str:
    """
    Pipeline principal do chatbot
    """
    try:
        if agent == "auto":
            agent, _ = run_router_agent(query, session_id)
            
        if agent == "tecnico":
            return run_rag_agent_tecnico(query, session_id)
            
        if agent == "faq":
            answer, _ = run_faq_agent(query, contexts)
            return answer
            
        # ... outros agents conforme necessário
        
        return "Desculpe, não entendi sua pergunta."
        
    except Exception as e:
        logger.exception("Erro no pipeline")
        return f"Erro ao processar: {str(e)}"


def run_assistente_agent(query: str, session_id: str | None = None, agent: str = "assistente", contexts: list | None = None) -> str:
    """
    Executa o agente "assistente".
    Aceita contextos opcionais para permitir o roteamento ao RAG organizacional quando apropriado.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Roteamento automático se solicitado (mantém compatibilidade)
    resolved_agent = agent
    if agent == "auto":
        try:
            resolved_agent, _ = run_router_agent(query, session_id)
        except Exception:
            resolved_agent = "assistente"

    # Se o roteador retornar "assistente" mas os contextos indicarem conteúdo organizacional, roteia para "organizacional"
    if resolved_agent == "assistente" and _contexts_match_query(contexts, query):
        logger.info("Assistente flow: CONTEXT indica que deve ser ORGANIZACIONAL")
        rag_output, rag_context = run_rag_agent_organizacional(query, session_id, contexts)
    else:
        # Inicializa memória conforme o agente resolvido
        if resolved_agent == "assistente":
            chat_message_history = get_memory_assistente(session_id)
        elif resolved_agent == "tecnico":
            chat_message_history = get_memory_tecnico(session_id)
        else:
            chat_message_history = None

        if chat_message_history:
            chat_message_history.add_user_message(query)

        # Etapa 1 - Verificação de guardrail
        guard_is_valid, guard_output = run_guardrail_agent(query, session_id)
        if not guard_is_valid:
            if chat_message_history:
                chat_message_history.add_ai_message(guard_output)
            return guard_output

        # Etapa 2 - Geração de resposta com RAG específico do agente
        if resolved_agent == "assistente":
            # Passa contextos se o RAG do assistente suportar
            try:
                rag_output, rag_context = run_rag_agent_assistente(query, session_id, contexts)  # prefer contextual call
            except TypeError:
                rag_output, rag_context = run_rag_agent_assistente(query, session_id)
        elif resolved_agent == "tecnico":
            rag_output, rag_context = run_rag_agent_tecnico(query, session_id)
        elif resolved_agent == "faq":
            rag_output, rag_context = run_faq_agent(query)
        else:
            # Agente inesperado: fallback para FAQ
            rag_output, rag_context = run_faq_agent(query)

    # Etapa 3 - Validação da resposta com o juiz
    try:
        judge_is_valid, judge_output = run_judge_agent(
            query, rag_output, rag_context, session_id, resolved_agent
        )
    except Exception:
        logger.exception("Judge agent failed; returning RAG output")
        judge_is_valid, judge_output = True, rag_output

    if judge_is_valid:
        # Armazena na memória, se existir
        try:
            if chat_message_history:
                chat_message_history.add_ai_message(rag_output)
        except Exception:
            logger.exception("falha ao salvar mensagem RAG no histórico")
        return rag_output
    else:
        try:
            if chat_message_history:
                chat_message_history.add_ai_message(judge_output)
        except Exception:
            logger.exception("falha ao salvar mensagem do juiz no histórico")
        return judge_output


if __name__ == "__main__":
    # CLI simples para executar o pipeline como módulo
    # Uso (a partir da raiz do repositório):
    #   python -m chat_bot.chat_real.sinara.pipeline --query "oi" --agent assistente --session test-1
    import argparse

    parser = argparse.ArgumentParser(description="Run Sinara pipeline once")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--agent", default="assistente", choices=["assistente", "tecnico", "organizacional", "auto"], help="Agent to use ('auto' usa o roteador)")
    parser.add_argument("--session", dest="session_id", default=None, help="Session ID (optional)")
    args = parser.parse_args()

    out = run_pipeline(query=args.query, session_id=args.session_id, agent=args.agent)
    print(out)

