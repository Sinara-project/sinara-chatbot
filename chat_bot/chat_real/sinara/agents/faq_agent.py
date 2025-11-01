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


class AgentePerguntas:
    """Agente responsÃ¡vel por responder perguntas frequentes do sistema"""
    
    def __init__(self):
        """Inicializa o agente com modelo e prompt"""
        self.modelo = self._inicializar_modelo()
        self.prompt = self._criar_prompt()

    def _inicializar_modelo(self) -> Optional[ChatGoogleGenerativeAI]:
        """Inicializa o modelo de IA com configuraÃ§Ãµes do ambiente"""
        chave_api = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not chave_api:
            return None
            
        nome_modelo = (
            os.getenv("GEMINI_MODEL_FAQ")
            or os.getenv("GEMINI_CHAT_MODEL")
            or "gemini-1.5-flash-latest"
        )
        
        try:
            return ChatGoogleGenerativeAI(
                model=nome_modelo,
                google_api_key=chave_api,
                temperature=0.1,
            )
        except Exception:
            return None

    def _criar_prompt(self) -> ChatPromptTemplate:
        """Cria o template do prompt para o modelo"""
        return ChatPromptTemplate.from_messages([
            (
                "system",
                """VocÃª Ã© um especialista no sistema Sinara, focado em explicar suas funcionalidades.

IMPORTANTE:
1. Use SOMENTE as informaÃ§Ãµes do contexto fornecido
2. Se encontrar a informaÃ§Ã£o no contexto, forneÃ§a uma resposta direta e objetiva
3. Inclua detalhes especÃ­ficos mencionados no contexto
4. Se a informaÃ§Ã£o nÃ£o existir no contexto, responda apenas: "NÃ£o encontrei informaÃ§Ãµes especÃ­ficas sobre isso no sistema"
5. NÃ£o mencione o contexto na resposta, apenas use seu conteÃºdo
6. NÃ£o invente ou adicione informaÃ§Ãµes alÃ©m do contexto""",
            ),
            (
                "human",
                """CONTEXTO:
{context}

PERGUNTA:
{query}

Responda usando APENAS as informaÃ§Ãµes do contexto acima.""",
            ),
        ])

    def _analisar_contexto(self, ctx: str) -> Dict[str, str]:
        """Analisa e separa o contexto em tÃ­tulo, seÃ§Ã£o e conteÃºdo"""
        partes = ctx.split("\n")
        resultado = {"titulo": "", "secao": "", "conteudo": ""}
        if len(partes) >= 3:
            resultado["titulo"] = partes[0].strip()
            resultado["secao"] = partes[1].strip()
            resultado["conteudo"] = "\n".join(partes[2:]).strip()
        else:
            resultado["conteudo"] = ctx.strip()
        return resultado

    def _normalizar(self, s: str) -> str:
        """Normaliza texto removendo acentos e caracteres especiais"""
        s = s or ""
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return s.strip()

    def _extrair_contexto(self, contextos: List[str], pergunta: str) -> str:
        """Extrai o contexto mais relevante para a pergunta"""
        if not contextos:
            return ""
            
        termos_pergunta = set(self._normalizar(pergunta).split())
        melhor_pontuacao = 0
        melhor_contexto = contextos[0]
        
        for ctx in contextos:
            pontuacao = 0
            ctx_norm = self._normalizar(ctx)
            pontuacao += sum(2 for termo in termos_pergunta if termo in ctx_norm)
            
            if "\n" in ctx:
                titulo = ctx.split("\n")[0].lower()
                pontuacao += sum(3 for termo in termos_pergunta if termo in titulo)
                
            if pontuacao > melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhor_contexto = ctx
                
        return melhor_contexto

    def _buscar_resposta_direta(self, contextos: List[str], pergunta: str) -> Optional[str]:
        """Busca uma resposta direta nos contextos se houver match forte"""
        p = self._normalizar((pergunta or "").strip())
        if not p or not contextos:
            return None
            
        tokens = [t for t in p.split() if len(t) > 2]
        for ctx in contextos:
            partes = self._analisar_contexto(ctx)
            texto = self._normalizar(f"{partes['titulo']} {partes['secao']} {partes['conteudo']}")
            
            if p and p in texto:
                return (partes["conteudo"] or ctx).strip()
                
            if tokens:
                sobreposicao = sum(1 for t in tokens if t in texto)
                if sobreposicao >= max(1, int(len(tokens) * 0.6)):
                    return (partes["conteudo"] or ctx).strip()
        return None

    def gerar_resposta(self, pergunta: str, contextos: Optional[List[str]] = None) -> Tuple[str, str]:
        """
        Gera resposta usando contextos ou payload FAQ
        
        Args:
            pergunta: Texto da pergunta
            contextos: Lista opcional de contextos
            
        Returns:
            Tupla (resposta, contexto_usado)
        """
        try:
            # Resolve texto do contexto
            if contextos is None:
                payload = get_faq_context(pergunta)
                contexto = payload.get("context", "") if isinstance(payload, dict) else str(payload or "")
            else:
                if isinstance(contextos, list):
                    contexto = self._extrair_contexto(contextos, pergunta)
                elif isinstance(contextos, str):
                    contexto = contextos
                else:
                    contexto = ""

            logger.info(f"Processando pergunta FAQ: {pergunta}")
            logger.debug(f"Usando contextos: {contexto}")

            if not contexto:
                return "NÃ£o encontrei informaÃ§Ãµes especÃ­ficas sobre isso no sistema.", ""

            # Se temos lista explÃ­cita de contextos, tenta match direto primeiro
            if isinstance(contextos, list) and contextos:
                direta = self._buscar_resposta_direta(contextos, pergunta)
                if direta:
                    return direta, contexto

            # Se modelo nÃ£o estÃ¡ disponÃ­vel, retorna trecho determinÃ­stico do contexto
            if self.modelo is None:
                trecho = contexto.strip()
                return (trecho[:1200], contexto)

            chain = self.prompt | self.modelo
            saida = chain.invoke({"context": contexto, "query": pergunta})
            resposta = getattr(saida, "content", None) or str(saida)
            logger.info("Resposta gerada (truncada): %s", (resposta or "")[:200].replace("`n", " "))
            return resposta.strip(), contexto

        except Exception:
            logger.exception("Erro gerando resposta FAQ")
            try:
                return (contexto[:1200], contexto) if contexto else ("Desculpe, ocorreu um erro ao processar sua pergunta.", "")
            except Exception:
                return "Desculpe, ocorreu um erro ao processar sua pergunta.", ""


def executar_agente_perguntas(pergunta: str, contextos: Optional[List[str]] = None) -> Tuple[str, str]:
    """Ponto de entrada para o agente de perguntas (compat).

    Mantido para chamadas existentes que usam nomenclatura em PT.
    """
    agente = AgentePerguntas()
    return agente.gerar_resposta(pergunta, contextos)


# Compatibilidade com o restante do cÃ³digo que espera English API
class FAQAgent(AgentePerguntas):
    def generate_response(self, query: str, contexts: Optional[List[str]] = None) -> Tuple[str, str]:
        return super().gerar_resposta(query, contexts)


def run_faq_agent(query: str, contexts: Optional[List[str]] = None) -> Tuple[str, str]:
    agent = FAQAgent()
    return agent.generate_response(query, contexts)
