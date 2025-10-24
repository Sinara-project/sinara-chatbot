import uuid
from .services.memory_assistente import get_memory as get_memory_assistente
from .services.memory_tecnico import get_memory as get_memory_tecnico
from .agents.guardrail_agent import run_guardrail_agent
from .agents.judge_agent import run_judge_agent
from .agents.rag_agent_assistente import run_rag_agent_assistente
from .agents.rag_agent_tecnico import run_rag_agent_tecnico
from .agents.rag_agent_organizacional import run_rag_agent_organizacional

def run_pipeline(query, session_id, agent="assistente"):
    if session_id is None and agent != "organizacional":
        session_id = str(uuid.uuid4())

    # Inicializa o histórico do MongoDB para essa sessão, conforme o agente
    if agent == "assistente":
        chat_message_history = get_memory_assistente(session_id)
    elif agent == "tecnico":
        chat_message_history = get_memory_tecnico(session_id)
    else:  # organizacional não usa memória
        chat_message_history = None

    # Salva a mensagem do usuário (se houver histórico)
    if chat_message_history:
        chat_message_history.add_user_message(query)

    # Etapa 1 - Verificação com o guardrail
    guard_is_valid, guard_output = run_guardrail_agent(query, session_id)
    if not guard_is_valid:
        if chat_message_history:
            chat_message_history.add_ai_message(guard_output)
        return guard_output

    # Etapa 2 - Geração da resposta com RAG específico do agente
    if agent == "assistente":
        rag_output, rag_context = run_rag_agent_assistente(query, session_id)
    elif agent == "tecnico":
        rag_output, rag_context = run_rag_agent_tecnico(query, session_id)
    else:  # organizacional
        rag_output, rag_context = run_rag_agent_organizacional(query, session_id)

    # Etapa 3 - Validação da resposta com o juiz
    judge_is_valid, judge_output = run_judge_agent(query, rag_output, rag_context, session_id)
    if judge_is_valid:
        if chat_message_history:
            chat_message_history.add_ai_message(rag_output)
        return rag_output
    else:
        if chat_message_history:
            chat_message_history.add_ai_message(judge_output)
        return judge_output


if __name__ == "__main__":
    # Simple CLI to run the pipeline as a module
    # Usage (from repo root):
    #   python -m chat_bot.chat_real.sinara.pipeline --query "oi" --agent assistente --session test-1
    import argparse

    parser = argparse.ArgumentParser(description="Run Sinara pipeline once")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--agent", default="assistente", choices=["assistente", "tecnico", "organizacional"], help="Agent to use")
    parser.add_argument("--session", dest="session_id", default=None, help="Session ID (optional)")
    args = parser.parse_args()

    out = run_pipeline(query=args.query, session_id=args.session_id, agent=args.agent)
    print(out)
