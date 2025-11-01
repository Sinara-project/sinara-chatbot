from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path
import json
import numpy as np
import os
import time
import unicodedata
import re

"""
Serviço RAG (Retrieval-Augmented Generation)
Este módulo implementa a recuperação de contextos similares para consultas,
usando embeddings do Google AI quando disponível ou BM25 como fallback offline.
"""

# Cache para otimização de performance
_json_texts: list[str] = []  # Chunks de texto processados
_json_vecs: list[np.ndarray] | None = None  # Vetores de embedding correspondentes
_json_mtime: float | None = None  # Timestamp do arquivo para verificar mudanças

# Cache para busca offline (BM25)
_raw_docs: list[dict] | None = None  # Documentos originais do JSON
_doc_texts: list[str] | None = None  # Textos completos por documento
_doc_tokens: list[list[str]] | None = None  # Tokens por documento
_doc_lengths: list[int] | None = None  # Comprimento (em tokens) por documento
_avgdl: float | None = None  # Comprimento médio dos documentos
_df_map: dict[str, int] | None = None  # Mapa de frequência dos termos


def _normalize(s: str) -> str:
    """
    Normaliza texto removendo acentos e convertendo para minúsculas
    Usado para padronizar textos antes da comparação
    """
    s = s or ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return s.lower()


def _tokenize(s: str) -> list[str]:
    """
    Converte texto em lista de tokens (palavras normalizadas)
    Usado para indexação e busca
    """
    return re.findall(r"[a-z0-9]+", _normalize(s))


def _chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 150) -> list[str]:
    """
    Divide texto em chunks menores com sobreposição
    Permite processamento de textos longos mantendo contexto
    
    Args:
        text: Texto a ser dividido
        chunk_size: Tamanho máximo de cada chunk
        chunk_overlap: Quantidade de sobreposição entre chunks
    """
    if not text:
        return []
    text = str(text)
    chunk_size = max(200, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - chunk_overlap
    return chunks


def _build_offline_index(raw_list: list[dict]):
    """
    Constrói índice offline para busca BM25
    Usado quando embeddings não estão disponíveis
    
    Processa documentos calculando:
    - Textos completos
    - Tokens por documento
    - Estatísticas para BM25 (comprimentos, frequências)
    """
    global _raw_docs, _doc_texts, _doc_tokens, _doc_lengths, _avgdl, _df_map
    _raw_docs = raw_list
    _doc_texts = []
    _doc_tokens = []
    _doc_lengths = []
    df: dict[str, int] = {}
    for d in raw_list:
        if not isinstance(d, dict):
            continue
        # Concatena título, seção e conteúdo
        title = d.get("title") or d.get("titulo") or ""
        section = d.get("section") or d.get("secao") or ""
        content = d.get("content") or d.get("conteudo") or ""
        full = "\n".join(x for x in [title, section, content] if x)
        _doc_texts.append(full)
        # Tokeniza e calcula estatísticas
        toks = _tokenize(full)
        _doc_tokens.append(toks)
        _doc_lengths.append(len(toks))
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    _df_map = df
    _avgdl = (sum(_doc_lengths) / len(_doc_lengths)) if _doc_lengths else 0.0


def _bm25_scores(qtoks: list[str]) -> list[tuple[float, int]]:
    """
    Calcula scores BM25 para tokens da query
    BM25 é um algoritmo de ranking que considera:
    - Frequência do termo (TF)
    - Frequência inversa nos documentos (IDF)
    - Comprimento do documento
    
    Args:
        qtoks: Tokens da query
    Returns:
        Lista de (score, índice_documento)
    """
    if not _doc_tokens or _avgdl is None or _df_map is None:
        return []
    N = len(_doc_tokens)
    k1 = 1.5  # Parâmetro de saturação de termo
    b = 0.75  # Parâmetro de normalização de comprimento
    scores = [0.0] * N
    for i, toks in enumerate(_doc_tokens):
        if not toks:
            continue
        dl = _doc_lengths[i] or 1
        for q in qtoks:
            df = _df_map.get(q, 0)
            if df == 0:
                continue
            # Calcula IDF
            idf = float(np.log((N - df + 0.5) / (df + 0.5) + 1.0))
            # Calcula TF normalizado
            tf = toks.count(q)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * dl / (_avgdl or 1))
            scores[i] += idf * (tf * (k1 + 1)) / denom
    return [(scores[i], i) for i in range(N)]


def _ensure_loaded():
    """
    Garante que dados estão carregados e atualizados
    Recarrega se arquivo fonte foi modificado
    """
    global _json_texts, _json_vecs, _json_mtime
    base = Path(__file__).resolve().parents[1]
    ctx_path = base / "db_script" / "contexto.json"
    mtime = os.path.getmtime(ctx_path)
    if (not _json_texts) or (_json_mtime is None) or (mtime != _json_mtime):
        with open(ctx_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_list = data if isinstance(data, list) else []
        texts: list[str] = []
        for d in raw_list:
            if not isinstance(d, dict):
                continue
            title = d.get("title") or d.get("titulo") or ""
            section = d.get("section") or d.get("secao") or ""
            content = d.get("content") or d.get("conteudo") or ""
            full = "\n".join(x for x in [title, section, content] if x)
            if full:
                texts.append(full)
        chunks: list[str] = []
        for t in texts:
            chunks.extend(_chunk_text(t))
        _json_texts = chunks
        _json_vecs = None
        _json_mtime = mtime
        _build_offline_index(raw_list)


def retrieve_similar_context(query: str, top_k: int = 3) -> list[str]:
    """
    Recupera contextos similares à query usando embeddings ou BM25
    """
    _ensure_loaded()
    
    # Heurística: ampliar K para consultas de funcionalidades do sistema (ex.: "bater ponto", "login", "página", "perfil")
    try:
        qn = _normalize(query)
        qtokens = set(_tokenize(qn))
        widen_kw = {"ponto", "bater", "registro", "login", "pagina", "perfil", "dashboard", "notificacao", "sistema", "web", "mobile"}
        dyn_k = max(top_k, 8) if (qtokens & widen_kw) else top_k
    except Exception:
        dyn_k = top_k

    # Tenta usar embeddings
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key and _json_texts:
        try:
            # Inicializa embeddings
            emb = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key
            )
            
            # Gera embedding da query
            query_vec = np.asarray(emb.embed_query(query), dtype="float32").ravel()
            
            # Gera ou recupera embeddings dos textos
            global _json_vecs
            if _json_vecs is None:
                _json_vecs = []
                for t in _json_texts:
                    if not t:
                        continue
                    vec = np.asarray(emb.embed_query(t), dtype="float32").ravel()
                    _json_vecs.append(vec)
            
            # Calcula similaridades
            results = []
            for t, v in zip(_json_texts, _json_vecs):
                if not t:
                    continue
                    
                # Similaridade por cosseno
                norm_a = float(np.linalg.norm(query_vec))
                norm_b = float(np.linalg.norm(v))
                if norm_a and norm_b:
                    sim = float(np.dot(query_vec, v) / (norm_a * norm_b))
                    results.append((sim, t))
                    
            # Retorna top_k (dinâmico) mais similares
            if results:
                results.sort(key=lambda x: x[0], reverse=True)
                return [t for _, t in results[:dyn_k]]
                
        except Exception:
            pass
    
    # Fallback para BM25
    tokens = _tokenize(query)
    scores = _bm25_scores(tokens)
    scores.sort(key=lambda x: x[0], reverse=True)
    
    if not _doc_texts:
        return []
        
    return [_doc_texts[i] for _, i in scores[:dyn_k]]


def retrieve_similar_context_with_scores(query: str, top_k: int = 5):
    """
    Similar ao retrieve_similar_context, mas inclui scores
    Útil para debugging e ajuste fino do sistema
    
    Args:
        query: Texto da consulta
        top_k: Número de contextos a retornar
    
    Returns:
        Lista de tuplas (score, texto) ordenada por relevância
    """
    load_dotenv(override=True)
    _ensure_loaded()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    emb = None
    if api_key:
        try:
            emb = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key,
                transport="rest",
            )
        except Exception:
            emb = None

    if emb is not None and _json_texts:
        query_vec = np.asarray(emb.embed_query(query), dtype="float32").ravel()
        global _json_vecs
        if _json_vecs is None:
            _json_vecs = [
                np.asarray(emb.embed_query(t) if t else np.zeros_like(query_vec), dtype="float32").ravel()
                for t in _json_texts
            ]

        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na == 0.0 or nb == 0.0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        pairs = []
        for t, v in zip(_json_texts, _json_vecs):
            if not t:
                continue
            try:
                s = cosine_sim(query_vec, v)
                pairs.append((s, t))
            except Exception:
                continue
        pairs.sort(key=lambda x: x[0], reverse=True)
        k = max(1, int(top_k))
        return pairs[:k]

    # Fallback offline: BM25
    qtoks = _tokenize(query)
    bm = _bm25_scores(qtoks)
    bm.sort(key=lambda x: x[0], reverse=True)
    k = max(1, int(top_k))
    if not _doc_texts:
        return []
    return [(s, _doc_texts[i]) for s, i in bm[:k]]
