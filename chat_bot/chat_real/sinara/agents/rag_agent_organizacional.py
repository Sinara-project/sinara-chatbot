import os
import json
import logging
import re
from typing import Tuple, Optional, List, Iterable
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

from ..services.memory_tecnico import get_memory
from ..services.rag_service import retrieve_similar_context, retrieve_similar_context_with_scores


load_dotenv(override=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RAGAgent:
    def __init__(self):
        self.model = self._initialize_model()
        self.prompt = self._load_prompt()

    def _initialize_model(self) -> ChatGoogleGenerativeAI:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing API key for Google generative model")
        model_name = os.getenv("GEMINI_MODEL_ORG") or os.getenv("GEMINI_CHAT_MODEL", "gemini-pro")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.2)

    def _load_prompt(self) -> ChatPromptTemplate:
        base_path = os.path.dirname(__file__)
        system_path = os.path.join(base_path, "../prompts/rag/organizacional/system_prompt_organizacional.txt")
        fewshot_path = os.path.join(base_path, "../prompts/rag/organizacional/fewshot.json")

        with open(system_path, "r", encoding="utf-8") as f:
            system_text = f.read()

        with open(fewshot_path, "r", encoding="utf-8") as f:
            raw_shots = json.load(f)

        # converte fewshots para o formato esperado
        examples = []
        for ex in raw_shots:
            # Garantir chaves 'human' e 'ai'
            examples.append({"human": ex.get("human", ""), "ai": ex.get("ai", "")})

        example_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template("{human}"),
            AIMessagePromptTemplate.from_template("{ai}"),
        ])

        fewshots = FewShotChatMessagePromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
        )

        #prompt final
        return ChatPromptTemplate.from_messages([
            ("system", system_text),
            fewshots,
            MessagesPlaceholder("memory"),
            ("human", (
                "Contexto disponível:\n{context}\n\n"
                "Pergunta do usuário:\n{query}\n\n"
                "Instruções:\n"
                "1) Primeiro verifique se a resposta existe no contexto. Se existir, responda com base no contexto (use o texto do contexto, seja objetivo).\n"
                "2) Se não existir no contexto, responda usando seu conhecimento geral apenas se tiver confiança na resposta.\n"
                "3) Se não tiver certeza, informe que não há informação suficiente.\n"
            )),
        ])

    def _extract_title_and_content(self, ctx) -> Tuple[Optional[str], str]:
        """
        Aceita itens de contexto que são:
         - dicionário com chaves 'title'/'content'
         - string no formato "Título\nSeção\nConteúdo" ou apenas "Conteúdo"
        Retorna (titulo_ou_None, conteudo_string)
        """
        if isinstance(ctx, dict):
            title = ctx.get("title") or ctx.get("titulo")
            content = ctx.get("content") or ctx.get("conteudo") or ""
            return title, str(content).strip()
        if isinstance(ctx, str):
            parts = [p.strip() for p in ctx.splitlines() if p.strip()]
            if len(parts) >= 3:
                # primeiro = title, segundo = section, resto = content
                content = " ".join(parts[2:])
                title = parts[0]
                return title, content.strip()
            if len(parts) == 2:
                return parts[0], parts[1]
            return None, ctx.strip()
        return None, ""

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        return text.strip()

    def _find_matching_context(self, query: str, contexts: Iterable) -> Optional[str]:
        """
        Heurística: retorna o conteúdo do primeiro contexto que claramente responde à consulta.
        Critérios:
         - consulta (frase completa) aparece no conteúdo (sem distinção entre maiúsculas/minúsculas), ou
         - tokens da consulta atingem limite mínimo de presença no conteúdo.
        """
        if not contexts:
            return None
        q_norm = self._normalize(query)
        q_tokens = [t for t in q_norm.split() if len(t) > 2]

        for ctx in contexts:
            title, content = self._extract_title_and_content(ctx)
            if not content:
                continue
            content_norm = self._normalize(content)
            # frase completa
            if q_norm and q_norm in content_norm:
                logger.info("frase completa (title=%s)", title)
                return content
            # o token 
            hits = sum(1 for t in q_tokens if t in content_norm)
            if q_tokens and hits >= max(1, len(q_tokens) // 2):
                logger.info("context (title=%s) hits=%d/%d", title, hits, len(q_tokens))
                return content
        return None

    def _get_context(self, query: str) -> str:
        """
        Recupera e formata contextos quando não forem fornecidos externamente.
        """
        try:
            ctx = retrieve_similar_context(query, top_k=5)
            if not ctx:
                logger.debug("retrieve_similar_context retornou vazio")
                return ""
            formatted: List[str] = []
            for item in ctx:
                _, content = self._extract_title_and_content(item)
                if content:
                    formatted.append(content)
            joined = "\n\n".join(formatted)
            logger.info("Contextos recuperados: %d items", len(formatted))
            return joined
        except Exception as e:
            logger.exception("Erro ao recuperar contexto")
            return ""

    def generate_response(self, query: str, session_id: str, provided_contexts: list | None = None) -> Tuple[str, str]:
        """
        Primeiro verifica se algum contexto (fornecido ou recuperado) responde diretamente;
        se sim, retorna o texto do contexto. Caso contrário, chama o modelo.
        """
        # Preferir contextos fornecidos (do manipulador HTTP) se presentes
        contexts_to_check = provided_contexts if provided_contexts is not None else retrieve_similar_context(query, top_k=5)

        # Tentar correspondência direta de contexto
        try:
            matched = self._find_matching_context(query, contexts_to_check)
            if matched:
                # Retornar contexto literal (usuário queria texto preciso do contexto)
                logger.info("Respondendo diretamente com contexto encontrado.")
                # retorna (texto_resposta, contexto_usado)
                return matched, matched
        except Exception:
            logger.exception("Erro durante verificação direta de contexto")

        # Se não houver correspondência direta de contexto, continuar com o fluxo padrão
        context_str = "\n\n".join(
            (self._extract_title_and_content(c)[1] for c in contexts_to_check)
        ) if contexts_to_check else self._get_context(query)

        memory = get_memory(session_id)
        memory_messages = getattr(memory, "messages", []) if memory else []

        try:
            prompt_inputs = {
                "context": context_str or "Nenhum contexto encontrado.",
                "query": query,
                "memory": memory_messages,
            }
            logger.info("Invocando modelo organizacional. Query: %s", query)
            logger.debug("Prompt inputs keys: %s", list(prompt_inputs.keys()))

            chain = self.prompt | self.model
            output = chain.invoke(prompt_inputs)

            content = getattr(output, "content", None) or str(output)
            logger.info("Modelo retornou resposta (primeiros 300 chars): %s", content[:300].replace("\n", " "))
            return content, context_str
        except Exception:
            logger.exception("Erro ao gerar resposta com o modelo")
            return self._fallback_response(query, context_str)

    def _fallback_response(self, query: str, context: str) -> Tuple[str, str]:
        try:
            pairs = retrieve_similar_context_with_scores(query, top_k=3)
            if pairs:
                first = pairs[0]
                text = first[1] if isinstance(first, (list, tuple)) and len(first) >= 2 else (first if isinstance(first, str) else "")
                if text:
                    logger.info("Fallback retornando trecho do RAG service")
                    return (text.strip()[:1200], context)
        except Exception:
            logger.exception("Erro no fallback RAG")
        return ("Desculpe, não encontrei informações suficientes para responder com confiança.", context)


def run_rag_agent_organizacional(query: str, session_id: str, contexts: list | None = None) -> Tuple[str, str]:
    """
    Wrapper resiliente: se a inicialização do modelo falhar (ex: chave API ausente),
    retorna um trecho determinístico do contexto como alternativa.
    """
    try:
        agent = RAGAgent()
        logger.info("Iniciando agente: query=%s session_id=%s", query, session_id)
        if contexts:
            logger.debug("Usando contextos fornecidos: %s", contexts)
        return agent.generate_response(query, session_id, contexts)
    except Exception as e:
        logger.error("Falha ao inicializar agente organizacional: %s", e)
        # fallback determinístico
        try:
            ctx_list = retrieve_similar_context(query, top_k=5)
            context_joined = "\n\n".join(ctx_list) if isinstance(ctx_list, list) else str(ctx_list or "")
            pairs = retrieve_similar_context_with_scores(query, top_k=3)
            if pairs:
                top = pairs[0]
                text = top[1] if isinstance(top, (list, tuple)) and len(top) > 1 else (top if isinstance(top, str) else "")
                snippet = str(text).strip()
                if snippet:
                    return snippet[:1200], context_joined
            return ("Desculpe, nao encontrei informacoes suficientes para responder com confianca.", context_joined)
        except Exception:
            logger.exception("Erro no fallback do organizacional")
            return ("Desculpe, nao consegui processar sua pergunta agora.", "")
