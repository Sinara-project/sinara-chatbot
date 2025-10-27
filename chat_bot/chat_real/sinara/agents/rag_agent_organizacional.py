import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..services.memory_tecnico import get_memory
from ..services.rag_service import retrieve_similar_context, retrieve_pdf_context
from dotenv import load_dotenv

# Carrega variáveis e chave
load_dotenv(override=True)
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Defina GEMINI_API_KEY (ou GOOGLE_API_KEY) no .env/ambiente.")


def _get_chat_model(model_name: str):
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=API_KEY,
        temperature=0.3,
    )


# Lê o template do prompt
system_prompt_path = os.path.join(
    os.path.dirname(__file__), "../prompts/rag/organizacional/system_prompt_organizacional.txt"
)
with open(system_prompt_path, "r", encoding="utf-8") as x:
    system_text = x.read()
system_prompt = ("system", system_text)

# Lê exemplos few-shot
fewshots_path = os.path.join(
    os.path.dirname(__file__), "../prompts/rag/organizacional/fewshot.json"
)
with open(fewshots_path, "r", encoding="utf-8") as x:
    shots = json.load(x)

example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}")
])

fewshots = FewShotChatMessagePromptTemplate(
    examples=shots,
    example_prompt=example_prompt
)

# Monta prompt final (inclui histórico opcional e query)
rag_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    fewshots,
    MessagesPlaceholder("memory"),
    ("human", "Contexto:\n{context}\n\nPergunta:\n{query}")
])


def run_rag_agent_organizacional(query, session_id):
    try:
        # Recupera contexto do RAG (aumentei top_k para pegar mais trechos relevantes)
        context_list = retrieve_similar_context(query, top_k=5)
        # garante que context seja string concatenada para o template
        if isinstance(context_list, list):
            context = "\n".join(context_list)
        else:
            context = str(context_list or "")
    except Exception as e:
        print(f"Erro na recuperação de contexto: {e}")
        return "Desculpe, houve um problema ao processar sua pergunta. Tente novamente mais tarde.", ""

    # Tenta substituir por contexto do PDF (FAQ); mantém fallback caso falhe
    try:
        _pdf_ctx = retrieve_pdf_context(query, top_k=5)
        if _pdf_ctx:
            context = _pdf_ctx
    except Exception:
        pass

    try:
        memory = get_memory(session_id)
        memory_messages = getattr(memory, "messages", []) if memory is not None else []

        model = _get_chat_model(os.getenv("GEMINI_MODEL_ORG") or os.getenv("GEMINI_CHAT_MODEL", "gemini-pro"))

        # Prepara o payload corretamente usando texto do contexto e mensagens de memória
        prompt_payload = rag_prompt.format(context=context, query=query, memory=memory_messages)
        output = model.invoke(prompt_payload)
        content = getattr(output, "content", None) or str(output)
        return content, context
    except Exception as e:
        print(f"Erro na geração da resposta: {e}")

    return "Desculpe, houve um problema ao processar sua pergunta. Tente novamente mais tarde.", ""
