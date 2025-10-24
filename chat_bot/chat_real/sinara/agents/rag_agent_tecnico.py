import os, json
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..services.memory_tecnico import get_memory
from ..services.rag_service import retrieve_similar_context
from dotenv import load_dotenv

# Carregado .env para o processo ANTES de instanciar o LLM
load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Defina GEMINI_API_KEY (ou GOOGLE_API_KEY) no ambiente/.env antes de iniciar."
    )

# Conecta com o Gemini para  de respostas
def _get_chat_model(model_name: str):
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=API_KEY,
        temperature=0.3,
    )

# LÃª o template do prompt
system_prompt_path = os.path.join(
    os.path.dirname(__file__), "../prompts/rag/tecnico/system_prompt_tecnico.txt"
)
with open(system_prompt_path, "r", encoding="utf-8") as x:
    system_text = x.read()
system_prompt = ("system", system_text)

# Le xemplos few-shot
fewshots_path = os.path.join(
    os.path.dirname(__file__), "../prompts/rag/tecnico/fewshot.json"
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

# Monta prompt final (inclui histÃ³rico opcional e query)
rag_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    fewshots,
    MessagesPlaceholder("memory"),
    ("human", "Contexto:\n{context}\n\nPergunta:\n{query}")
])

def run_rag_agent_tecnico(query, session_id):
    try:
        # Recupera contexto do RAG
        context = retrieve_similar_context(query)
    except Exception as e:
        print(f"Erro na recuperaÃ§Ã£o de contexto: {e}")
        return "Desculpe, houve um problema ao processar sua pergunta. Tente novamente mais tarde.", ""

    try:
        memory = get_memory(session_id)
        model = _get_chat_model(os.getenv("GEMINI_MODEL_TECNICO") or os.getenv("GEMINI_CHAT_MODEL", "gemini-1.0-pro"))
        output = model.invoke(rag_prompt.format(context=context, query=query, memory=memory.messages))
        return output.content, context
    except Exception as e:
        print(f"Erro na geraÃ§Ã£o da resposta: {e}")

    return "Desculpe, houve um problema ao processar sua pergunta. Tente novamente mais tarde.",





