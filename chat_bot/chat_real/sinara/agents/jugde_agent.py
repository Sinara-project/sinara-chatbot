import os, json
from typing import Union
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from pymongo import MongoClient

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..services.memory_tecnico import get_memory

from dotenv import load_dotenv

# Carrega variáveis do .env para o processo ANTES de instanciar o LLM
load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Defina GEMINI_API_KEY (ou GOOGLE_API_KEY) no ambiente/.env antes de iniciar."
    )


class JudgeOutput(BaseModel):
    flag: int = Field(
        description='0 se a entrada for válida, 1 se for ofensiva'
    )
    message: Union[str, None] = Field(
        description='Mensagem educada para fugir do assunto caso flag=1, ou None se flag=0'
    )

# Conecta com o Gemini para geração de respostas
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
).with_structured_output(JudgeOutput)

# Lê o template do prompt
system_prompt_path = os.path.join(
    os.path.dirname(__file__), "../prompts/judge/system_prompt.txt"
)
with open(system_prompt_path, "r", encoding="utf-8") as x:
    system_text = x.read()
system_prompt = ("system", system_text)

# Lê exemplos few-shot
fewshots_path = os.path.join(
    os.path.dirname(__file__), "../prompts/judge/fewshot.json"
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
judge_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    fewshots,
    MessagesPlaceholder("memory"),
    ("human",
     "Contexto:\n{context}\n\nResposta do RAG:\n{rag_output}\n\nPergunta do usuário:\n{query}")
])

# Declara a pipeline
pipeline = judge_prompt | model

# Ajuste robusto de variáveis e caminho do arquivo
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_db_name = os.getenv("MONGO_DB", "eta_app")
mongo_client = MongoClient(mongo_uri)
mongo_db = mongo_client[mongo_db_name]
contexto_collection = mongo_db["contexto"]

# Lê os dados do contexto.json (pasta correta = db_script)
with open("db_script/contexto.json", "r", encoding="utf-8") as f:
    contexto_docs = json.load(f)

def run_judge_agent(query, rag_output, context, session_id):
    try:
        memory = get_memory(session_id)
        output: JudgeOutput = pipeline.invoke({
            "query": query,
            "rag_output": rag_output,
            "context": context,
            "memory": memory.messages
        })

        if output.flag == 0:
            return True, None
        else:
            return False, output.message

    except Exception as e:
        print(f"Erro na geração da resposta: {e}")

    # Retornar sempre (mensagem, contexto) — contexto vazio se não houver
    return "Desculpe, houve um problema ao processar sua pergunta.", ""

# Código para leitura dos arquivos de perguntas e gravação das respostas
test_type = "tipo_exemplo"  # Substitua pelo tipo de teste desejado

# pasta real: teste/perguntas
with open(f'chat_real/sinara/teste/perguntas/{test_type}_pergunta.json', 'r', encoding='utf-8') as file:
    questions = json.load(file)

answers = []
for question in questions:
    query = question["query"]
    rag_output = question["rag_output"]
    context = contexto_docs.get(query, "")  # Obtém o contexto correspondente à pergunta
    session_id = question.get("session_id", "default_session")

    is_valid, message = run_judge_agent(query, rag_output, context, session_id)

    answer = {
        "query": query,
        "rag_output": rag_output,
        "context": context,
        "session_id": session_id,
        "is_valid": is_valid
    }

    if not is_valid:
        answer["offensive_message"] = message

    answers.append(answer)

# pasta real: teste/respostas
with open(f"chat_real/sinara/teste/respostas/{test_type}_resposta.json", "w", encoding="utf-8") as f:
    json.dump(answers, f, indent=2, ensure_ascii=False)