import os, json, logging
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..services.memory_assistente import get_memory
from ..services.rag_service import retrieve_similar_context, retrieve_pdf_context

from dotenv import load_dotenv

# Carrega variáveis do .env para o processo ANTES de instanciar o LLM
load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Defina GEMINI_API_KEY (ou GOOGLE_API_KEY) no ambiente/.env antes de iniciar."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conecta com o Gemini para geração de respostas
def _get_chat_model(model_name: str):
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=API_KEY,
        temperature=0.3,
    )



# Modelos fallback conhecidos (ordem de preferência)
FALLBACK_MODELS = [
    "gemini-pro",
    "gemini-1.0-pro",
]

# Lê o template do prompt
system_prompt_path = os.path.join(
    os.path.dirname(__file__), "../prompts/rag/assistente/system_prompt_assistente.txt"
)
with open(system_prompt_path, "r", encoding="utf-8") as x:
    system_text = x.read()
system_prompt = ("system", system_text)

# Lê exemplos few-shot
fewshots_path = os.path.join(
    os.path.dirname(__file__), "../prompts/rag/assistente/fewshot.json"
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

def _try_models_and_invoke(prompt_payload):
    """
    Tenta instanciar/invocar modelos em sequência até obter sucesso.
    prompt_payload: dicionário com keys necessárias para formatar/invocar o prompt.
    Retorna (response_content, used_model) ou (None, None) em falha.
    """
    env_model = os.getenv("GEMINI_MODEL_ASSISTENTE") or os.getenv("GEMINI_CHAT_MODEL")
    candidates = []
    if env_model:
        candidates.append(env_model)
    candidates.extend(FALLBACK_MODELS)

    for candidate in candidates:
        try:
            logger.info(f"Tentando modelo: {candidate}")
            model = _get_chat_model(candidate)
            output = model.invoke(prompt_payload)
            # output pode ser objeto com .content (compatível com uso anterior)
            content = getattr(output, "content", None) or str(output)
            return content, candidate
        except Exception as e:
            logger.warning(f"Modelo {candidate} falhou: {e}")
            # tenta próximo
            continue

    return None, None

def run_rag_agent_assistente(query, session_id):
    try:
        # Prioriza contexto do PDF (FAQ); fallback para Mongo/FAISS
        try:
            context = retrieve_pdf_context(query, top_k=6)
        except Exception:
            context = retrieve_similar_context(query)
    except Exception as e:
        logger.error(f"Erro na recuperação de contexto: {e}")
        return "Desculpe, houve um problema ao processar sua pergunta. Tente novamente mais tarde.", ""

    try:
        memory = get_memory(session_id)
        # Prepara payload para o prompt (mantendo compatibilidade com formatação original)
        memory_messages = getattr(memory, "messages", [])
        prompt_payload = rag_prompt.format(context=context, query=query, memory=memory_messages)

        response_content, used_model = _try_models_and_invoke(prompt_payload)
        if response_content is not None:
            return response_content, context

        logger.error("Nenhum modelo disponível respondeu com sucesso.")
        return "Desculpe, não foi possível gerar a resposta no momento. Tente novamente mais tarde.", ""

    except Exception as e:
        logger.error(f"Erro na geração da resposta: {e}")
        return "Desculpe, houve um problema ao processar sua pergunta. Tente novamente mais tarde.", ""





