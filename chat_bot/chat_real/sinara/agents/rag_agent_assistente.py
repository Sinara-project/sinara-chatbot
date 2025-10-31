import os
import json
import logging
from dotenv import load_dotenv
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

from ..services.memory_assistente import get_memory
from ..services.rag_service import retrieve_similar_context, retrieve_similar_context_with_scores


load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_chat_model(model_name: str):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("API key ausente")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3,
    )


FALLBACK_MODELS = [
    "gemini-pro",
    "gemini-1.0-pro",
]


system_prompt_path = os.path.join(
    os.path.dirname(__file__), "../prompts/rag/assistente/system_prompt_assistente.txt"
)
with open(system_prompt_path, "r", encoding="utf-8") as x:
    system_text = x.read()
system_prompt = ("system", system_text)

fewshots_path = os.path.join(
    os.path.dirname(__file__), "../prompts/rag/assistente/fewshot.json"
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

rag_prompt = ChatPromptTemplate.from_messages(
    [
        system_prompt,
        fewshots,
        MessagesPlaceholder("memory"),
        ("human", "Contexto:\n{context}\n\nPergunta:\n{query}"),
    ]
)


def run_rag_agent_assistente(query, session_id):
    try:
        ctx = retrieve_similar_context(query)
        context = "\n".join(ctx) if isinstance(ctx, list) else str(ctx or "")
    except Exception as e:
        logger.error(f"Erro na recuperação de contexto: {e}")
        return (
            "Desculpe, houve um problema ao processar sua pergunta. Tente novamente mais tarde.",
            "",
        )

    try:
        memory = get_memory(session_id)
        env_model = os.getenv("GEMINI_MODEL_ASSISTENTE") or os.getenv("GEMINI_CHAT_MODEL")
        candidates = [m for m in [env_model, *FALLBACK_MODELS] if m]

        for candidate in candidates:
            try:
                logger.info(f"Tentando modelo: {candidate}")
                model = _get_chat_model(candidate)
                chain = rag_prompt | model
                output = chain.invoke(
                    {
                        "context": context,
                        "query": query,
                        "memory": getattr(memory, "messages", []),
                    }
                )
                content = getattr(output, "content", None) or str(output)
                return content, context
            except Exception as e:
                logger.warning(f"Modelo {candidate} falhou: {e}")
                continue

        logger.error("Nenhum modelo disponível respondeu com sucesso.")
        # Fallback baseado na melhor correspondÃªncia
        pairs = retrieve_similar_context_with_scores(query, top_k=3)
        if pairs:
            _score, text = pairs[0]           
            return (str(text).strip()[:1200], context)
        return ("Não encontrei essainformação no nosso FAQ.", context)

    except Exception as e:
        logger.error(f"Erro na geração da resposta: {e}")
        pairs = retrieve_similar_context_with_scores(query, top_k=3)
        if pairs:
            _score, text = pairs[0]
            return (str(text).strip()[:1200], context)
        return ("Nãoo encontrei essa informação no nosso FAQ.", context)

    except Exception as e:
        logger.error(f"Erro na geração da resposta: {e}")
        pairs = retrieve_similar_context_with_scores(query, top_k=3)
        if pairs:
            _score, text = pairs[0]
            return (str(text).strip()[:1200], context)
        return ("Nãoo encontrei essa informação no nosso FAQ.", context)
