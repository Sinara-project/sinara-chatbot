import os
import json
from typing import Union, Tuple, Optional

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

from ..services.memory_tecnico import get_memory as get_memory_tecnico
from ..services.memory_assistente import get_memory as get_memory_assistente


load_dotenv(override=True)


class JudgeOutput(BaseModel):
    flag: int = Field(description="0 se a entrada for válida, 1 se for ofensiva")
    message: Union[str, None] = Field(
        description="Mensagem educada para fugir do assunto caso flag=1, ou None se flag=0"
    )


def build_pipeline(model_name: Optional[str] = None):
    chat_model = model_name or os.getenv("GEMINI_MODEL_JUDGE") or os.getenv(
        "GEMINI_CHAT_MODEL", "gemini-1.0-pro"
    )
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("API key ausente")
    model = ChatGoogleGenerativeAI(
        model=chat_model,
        google_api_key=api_key,
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
                "Contexto:\n{context}\n\nResposta do RAG:\n{rag_output}\n\nPergunta do usuário:\n{query}",
            ),
        ]
    )

    return judge_prompt | model


def run_judge_agent(
    query: str,
    rag_output: str,
    context: str,
    session_id: str,
    agent: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    try:
        if agent == "assistente":
            memory = get_memory_assistente(session_id)
        elif agent == "tecnico":
            memory = get_memory_tecnico(session_id)
        else:
            memory = None
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
        # Em falha do juiz, não bloqueia a resposta do RAG
        print(f"Erro na geração da resposta do juiz: {e}")
        return True, None

