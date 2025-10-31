from typing import Any, Dict, List, Tuple

from .rag_service import (
    retrieve_similar_context,
    retrieve_similar_context_with_scores,
)


def get_faq_context(
    question: str,
    k: int = 6,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Dict[str, Any]:
    """
    Recupera contexto do FAQ (arquivo JSON local) para uma pergunta.
    Retorna dict com a pergunta, matches (score, trecho) e o contexto concatenado.
    """
    if not question or not str(question).strip():
        raise ValueError("question deve ser uma string não-vazia")

    pairs: List[Tuple[float, str]] = retrieve_similar_context_with_scores(question, top_k=k)
    context_texts = [t for _s, t in pairs]
    context_joined = "\n\n---\n\n".join(context_texts)

    return {
        "question": question,
        "k": k,
        "matches": pairs,
        "context": context_joined,
    }

