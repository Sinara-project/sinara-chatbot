import os
import logging
import re
import unicodedata
import difflib
from typing import Tuple, List, Optional, Dict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from ..services.faq_tool import get_faq_context
from ..services.rag_service import retrieve_similar_context

load_dotenv(override=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FAQAgent:
    def __init__(self):
        self.model = self._initialize_model()
        self.prompt = self._create_prompt()

    def _initialize_model(self) -> Optional[ChatGoogleGenerativeAI]:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        model_name = (
            os.getenv("GEMINI_MODEL_FAQ")
            or os.getenv("GEMINI_CHAT_MODEL")
            or "gemini-1.5-flash-latest"
        )
        try:
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.1,
            )
        except Exception:
            return None

    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            (
                "system",
                """Você é um especialista no sistema Sinara, focado em explicar suas funcionalidades.

IMPORTANTE:
1. Use SOMENTE as informações do contexto fornecido
2. Se encontrar a informação no contexto, forneça uma resposta direta e objetiva
3. Inclua detalhes específicos mencionados no contexto
4. Se a informação não existir no contexto, responda apenas: "Não encontrei informações específicas sobre isso no sistema"
5. Não mencione o contexto na resposta, apenas use seu conteúdo
6. Não invente ou adicione informações além do contexto""",
            ),
            (
                "human",
                """CONTEXTO:
{context}

PERGUNTA:
{query}

Responda usando APENAS as informações do contexto acima.""",
            ),
        ])

    def _parse_context(self, ctx: str) -> Dict[str, str]:
        parts = ctx.split("\n")
        result = {"title": "", "section": "", "content": ""}
        if len(parts) >= 3:
            result["title"] = parts[0].strip()
            result["section"] = parts[1].strip()
            result["content"] = "\n".join(parts[2:]).strip()
        else:
            result["content"] = ctx.strip()
        return result

    def _normalize(self, s: str) -> str:
        s = s or ""
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return s.strip()

    def _extract_context(self, contexts: List[str], query: str) -> str:
        """Extrai contexto mais relevante"""
        if not contexts:
            return ""
            
        query_terms = set(self._normalize(query).split())
        best_score = 0
        best_context = contexts[0]
        
        for ctx in contexts:
            score = 0
            
            # Score básico por termos
            ctx_norm = self._normalize(ctx)
            score += sum(2 for term in query_terms if term in ctx_norm)
            
            # Score por título
            if "\n" in ctx:
                title = ctx.split("\n")[0].lower()
                score += sum(3 for term in query_terms if term in title)
                
            if score > best_score:
                best_score = score
                best_context = ctx
                
        return best_context

    def _direct_match_answer(self, contexts: List[str], query: str) -> Optional[str]:
        """Return a direct answer from provided contexts if there's a strong match."""
        q = self._normalize((query or "").strip())
        if not q or not contexts:
            return None
        tokens = [t for t in q.split() if len(t) > 2]
        for ctx in contexts:
            parts = self._parse_context(ctx)
            text = self._normalize(f"{parts['title']} {parts['section']} {parts['content']}")
            if q and q in text:
                return (parts["content"] or ctx).strip()
            if tokens:
                overlap = sum(1 for t in tokens if t in text)
                if overlap >= max(1, int(len(tokens) * 0.6)):
                    return (parts["content"] or ctx).strip()
        return None

    def generate_response(self, query: str, contexts: Optional[List[str]] = None) -> Tuple[str, str]:
        """Generate response using contexts or the FAQ payload."""
        try:
            # Resolve context text
            if contexts is None:
                payload = get_faq_context(query)
                context = payload.get("context", "") if isinstance(payload, dict) else str(payload or "")
            else:
                if isinstance(contexts, list):
                    context = self._extract_context(contexts, query)
                elif isinstance(contexts, str):
                    context = contexts
                else:
                    context = ""

            logger.info("Processing FAQ query: %s", query)
            logger.debug("Using contexts: %s", context)

            if not context:
                return "Não encontrei informações específicas sobre isso no sistema.", ""

            # If we have explicit contexts list, try direct match first
            if isinstance(contexts, list) and contexts:
                direct = self._direct_match_answer(contexts, query)
                if direct:
                    return direct, context

                # Page-intent widening: if query resembles a specific page, widen retrieval
                qn = self._normalize((query or "").strip())
                if "pagina" in qn and ("inicial" in qn or "perfil" in qn or "gestao" in qn or "operador" in qn):
                    try:
                        extra = retrieve_similar_context(query, top_k=12)
                        if isinstance(extra, list) and extra:
                            wider = contexts + [c for c in extra if isinstance(c, str)]
                            new_context = self._extract_context(wider, query)
                            if new_context:
                                direct2 = self._direct_match_answer(wider, query)
                                if direct2:
                                    return direct2, new_context
                                # fallback to new_context if no direct answer
                                context = new_context
                    except Exception:
                        pass

            # If model is unavailable, return a deterministic snippet from context
            if self.model is None:
                snippet = context.strip()
                return (snippet[:1200], context)

            chain = self.prompt | self.model
            output = chain.invoke({"context": context, "query": query})
            response = getattr(output, "content", None) or str(output)
            logger.info("Generated response (truncated): %s", response[:200].replace("\n", " "))
            return response.strip(), context

        except Exception:
            logger.exception("Error generating FAQ response")
            # Best-effort fallback to context snippet instead of generic error
            try:
                return (context[:1200], context) if context else ("Desculpe, ocorreu um erro ao processar sua pergunta.", "")
            except Exception:
                return "Desculpe, ocorreu um erro ao processar sua pergunta.", ""


def run_faq_agent(query: str, contexts: Optional[List[str]] = None) -> Tuple[str, str]:
    """Entry point for FAQ agent"""
    agent = FAQAgent()
    return agent.generate_response(query, contexts)
