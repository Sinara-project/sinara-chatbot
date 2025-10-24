import os
import json
from typing import Union, Tuple, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from ..services.memory_tecnico import get_memory

from dotenv import load_dotenv

# Carrega variaveis do .env para o processo
load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Defina GEMINI_API_KEY (ou GOOGLE_API_KEY) no ambiente/.env antes de iniciar."
    )


class JudgeOutput(BaseModel):
    flag: int = Field(description="0 se a entrada for vÃ¡lida, 1 se for ofensiva")
    message: Union[str, None] = Field(
        description="Mensagem educada para fugir do assunto caso flag=1, ou None se flag=0"
    )


# Função para montar o pipeline (instancia LLM de forma explícita)
def build_pipeline(model_name: Optional[str] = None):
    chat_model = model_name or os.getenv("GEMINI_MODEL_JUDGE") or os.getenv("GEMINI_CHAT_MODEL", "gemini-1.0-pro")
    model = ChatGoogleGenerativeAI(
        model=chat_model,
        google_api_key=API_KEY,
    ).with_structured_output(JudgeOutput)
    base_dir = os.path.dirname(__file__)
    system_prompt_path = os.path.normpath(
        os.path.join(base_dir, "..", "prompts", "judge", "system_prompt.txt")
    )
    with open(system_prompt_path, "r", encoding="utf-8") as x:
        system_text = x.read()
    system_prompt = ("system", system_text)

    fewshots_path = os.path.normpath(
        os.path.join(base_dir, "..", "prompts", "judge", "fewshot.json")
    )
    with open(fewshots_path, "r", encoding="utf-8") as x:
        shots = json.load(x)

    example_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{human}"),
            AIMessagePromptTemplate.from_template("{ai}"),
        ]
    )

    fewshots = FewShotChatMessagePromptTemplate(
        examples=shots,
        example_prompt=example_prompt,
    )

    judge_prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            fewshots,
            MessagesPlaceholder("memory"),
            (
                "human",
                "Contexto:\n{context}\n\nResposta do RAG:\n{rag_output}\n\nPergunta do usuÃ¡rio:\n{query}",
            ),
        ]
    )

    return judge_prompt | model


# Leitura opcional de contexto para testes locais
_base_dir = os.path.dirname(__file__)
_contexto_docs = {}
_contexto_path = os.path.normpath(os.path.join(_base_dir, "..", "db_script", "contexto.json"))
if os.path.exists(_contexto_path):
    try:
        with open(_contexto_path, "r", encoding="utf-8") as f:
            _contexto_docs = json.load(f)
    except Exception:
        _contexto_docs = {}


def run_judge_agent(query: str, rag_output: str, context: str, session_id: str) -> Tuple[bool, Optional[str]]:
    """Executa o agente juiz para validar a resposta gerada pelo RAG.

    Retorna:
      - (True, None) se a resposta for vÃ¡lida
      - (False, mensagem) se a resposta for invÃ¡lida/ofensiva
    """
    try:
        memory = get_memory(session_id)
        pipeline = build_pipeline()
        output: JudgeOutput = pipeline.invoke(
            {
                "query": query,
                "rag_output": rag_output,
                "context": context,
                "memory": getattr(memory, "messages", []),
            }
        )

        if getattr(output, "flag", 1) == 0:
            return True, None
        return False, getattr(output, "message", None)

    except Exception as e:
        print(f"Erro na geraÃ§Ã£o da resposta: {e}")
        return False, "Desculpe, houve um problema ao processar sua pergunta."


if __name__ == "__main__":
    test_type = "tipo_exemplo"  # Substitua pelo tipo de teste desejado

    perguntas_path = os.path.normpath(
        os.path.join(_base_dir, "..", "teste", "perguntas", f"{test_type}_pergunta.json")
    )
    respostas_path = os.path.normpath(
        os.path.join(_base_dir, "..", "teste", "respostas", f"{test_type}_resposta.json")
    )

    if os.path.exists(perguntas_path):
        with open(perguntas_path, "r", encoding="utf-8") as file:
            questions = json.load(file)

        answers = []
        for question in questions:
            query = question.get("query")
            rag_output = question.get("rag_output")
            context = _contexto_docs.get(query, "")
            session_id = question.get("session_id", "default_session")

            is_valid, message = run_judge_agent(query, rag_output, context, session_id)

            answer = {
                "query": query,
                "rag_output": rag_output,
                "context": context,
                "session_id": session_id,
                "is_valid": is_valid,
            }

            if not is_valid:
                answer["offensive_message"] = message

            answers.append(answer)

        os.makedirs(os.path.dirname(respostas_path), exist_ok=True)
        with open(respostas_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=2, ensure_ascii=False)





