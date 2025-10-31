import os
import json
import logging
from typing import Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

from ..services.memory_tecnico import get_memory


def _load_api_key() -> str | None:
    load_dotenv(override=True)
    api = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api:
        return api
    local_env = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if os.path.exists(local_env):
        load_dotenv(dotenv_path=local_env, override=True)
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# API key carregada sob demanda em _get_chat_model


class GuardrailOutput(BaseModel):
    flag: int = Field(description="0 se a entrada for válida, 1 se for ofensiva")
    message: Union[str, None] = Field(
        description="Mensagem educada para fugir do assunto caso flag=1, ou None se flag=0"
    )


# Lê o template do prompt
system_prompt_path = os.path.join(
    os.path.dirname(__file__), "../prompts/guardrail/system_prompt.txt"
)
with open(system_prompt_path, "r", encoding="utf-8") as x:
    system_text = x.read()
system_prompt = ("system", system_text)

# Lê exemplos few-shot
fewshots_path = os.path.join(
    os.path.dirname(__file__), "../prompts/guardrail/fewshot.json"
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

# Monta prompt final (inclui histórico opcional e query)
guardrail_prompt = ChatPromptTemplate.from_messages(
    [
        system_prompt,
        fewshots,
        MessagesPlaceholder("memory"),
        ("human", "{query}"),
    ]
)


# Conecta com o Gemini para geração de respostas (instanciado sob demanda)
def _get_chat_model(model_name: str):
    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError("API key ausente")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
    ).with_structured_output(GuardrailOutput)


def run_guardrail_agent(query: str, session_id: str):
    try:
        memory = get_memory(session_id)

        preferred = os.getenv("GEMINI_MODEL_GUARDRAIL") or os.getenv(
            "GEMINI_CHAT_MODEL", "gemini-1.5-flash-latest"
        )
        candidates = []
        for m in [
            preferred,
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro",
        ]:
            if m and m not in candidates:
                candidates.append(m)

        for m in candidates:
            try:
                model = _get_chat_model(m)
                pipeline = guardrail_prompt | model
                output = pipeline.invoke(
                    {"query": query, "memory": getattr(memory, "messages", [])}
                )
                if getattr(output, "flag", 1) == 0:
                    return True, None
                else:
                    return False, getattr(output, "message", None)
            except Exception as e:
                msg = str(e)
                if ("NotFound" in msg) or ("is not found" in msg):
                    continue
                raise

    except Exception as e:
        print(f"Erro no guardrail: {e}")

    # Pass-through seguro: se não foi possível avaliar o guardrail,
    # não bloqueia a conversa; deixa os próximos estágios decidirem.
    return True, None
