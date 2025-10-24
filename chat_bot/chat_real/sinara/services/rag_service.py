from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ConfigurationError, ServerSelectionTimeoutError
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path
import json
import numpy as np
import faiss
import os

# Cache em memória para FAISS
_faiss_index = None
_faiss_texts = []
_faiss_dim = None
_json_texts = []
_json_vecs = None

def _ensure_faiss_index(mongo_uri: str, mongo_db: str):
    global _faiss_index, _faiss_texts, _faiss_dim
    if _faiss_index is not None:
        return
    client = None
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        coll = client[mongo_db]["contexto"]
        docs = list(coll.find({"embedding": {"$exists": True}}))
        if not docs:
            return
        embeddings = []
        texts = []
        for d in docs:
            emb = d.get("embedding")
            content = d.get("content") or d.get("conteudo")
            if not isinstance(emb, (list, tuple)) or content is None:
                continue
            try:
                vec = np.asarray(emb, dtype="float32").ravel()
            except Exception:
                continue
            embeddings.append(vec)
            texts.append(content)
        if not embeddings:
            return
        mat = np.vstack(embeddings).astype("float32")
        # Normaliza para usar similaridade de cosseno via produto interno
        faiss.normalize_L2(mat)
        dim = mat.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(mat)
        _faiss_index = index
        _faiss_texts = texts
        _faiss_dim = dim
    except (ConfigurationError, ServerSelectionTimeoutError, PyMongoError):
        # Falha ao conectar/carregar embeddings do Mongo: segue sem FAISS
        pass
    finally:
        try:
            if client is not None:
                client.close()
        except Exception:
            pass

def retrieve_similar_context(query: str, top_k: int = 3):
    """
    Retorna os 'top_k' conteúdos mais similares ao 'query' a partir da coleção MongoDB 'contexto'.
    Requisitos:
      - Variáveis: GEMINI_API_KEY (ou GOOGLE_API_KEY), MONGO_URI, MONGO_DB
      - Cada doc em 'contexto' deve ter: 'content' (str) e 'embedding' (list[float])
    """
    load_dotenv(override=True)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Defina GEMINI_API_KEY (ou GOOGLE_API_KEY) no .env/ambiente.")

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db  = os.getenv("MONGO_DB", "DB_Sinara")

    # Tenta conectar ao Mongo, mas nao deixa a API travar caso falhe (ex.: DNS SRV bloqueado)
    client = None
    collection = None
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        collection = client[mongo_db]["contexto"]
    except (ConfigurationError, ServerSelectionTimeoutError, PyMongoError):
        # Sem conexao com Mongo; seguiremos com FAISS se ja carregado, caso contrario retorna []
        pass

    # Embeddings do Gemini (mesma API do embedding)
    # Use a valid embedding model name for the Google Generative AI API
    # For embeddings, the API expects full resource format "models/text-embedding-004"
    emb = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
        transport="rest",  # evita grpc.aio e a necessidade de event loop
    )

    # Garante índice FAISS carregado (usa embeddings salvos no Mongo)
    try:
        _ensure_faiss_index(mongo_uri, mongo_db)
    except Exception:
        pass

    # Vetor da consulta
    query_vec = np.asarray(emb.embed_query(query), dtype="float32").ravel()

    def cosine_sim(a: np.ndarray, b) -> float:
        b = np.asarray(b, dtype=float).ravel()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    results = []
    top_k = max(1, int(top_k))
    # Se FAISS disponível, usa busca vetorial eficiente
    if _faiss_index is not None and len(_faiss_texts) > 0:
        q = query_vec[None, :].astype("float32")
        faiss.normalize_L2(q)
        _, idxs = _faiss_index.search(q, min(top_k, len(_faiss_texts)))
        results = [_faiss_texts[i] for i in idxs[0] if 0 <= i < len(_faiss_texts)]
    else:
        # Fallback: varre e calcula cosseno em Python
        score_docs = []
        if collection is not None:
            cursor = collection.find({"embedding": {"$exists": True}})
            for doc in cursor:
                emb_list = doc.get("embedding")
                content  = doc.get("content") or doc.get("conteudo") or ""
                if not isinstance(emb_list, (list, tuple)) or not content:
                    continue
                try:
                    score = cosine_sim(query_vec, emb_list)
                except Exception:
                    continue
                score_docs.append((score, content))
            score_docs.sort(key=lambda x: x[0], reverse=True)
            results = [content for score, content in score_docs[:top_k]]
        else:
            # Fallback local: usa arquivo contexto.json se Mongo/FAISS indisponiveis
            global _json_texts, _json_vecs
            try:
                if not _json_texts:
                    base = Path(__file__).resolve().parents[1]
                    ctx_path = base / "db_script" / "contexto.json"
                    with open(ctx_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    _json_texts = [
                        (d.get("content") or d.get("conteudo") or "") for d in data if isinstance(d, dict)
                    ]
                if _json_vecs is None:
                    # calcula embeddings locais uma vez
                    _json_vecs = [
                        np.asarray(emb.embed_query(t) if t else np.zeros_like(query_vec), dtype="float32").ravel()
                        for t in _json_texts
                    ]
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
                results = [t for s, t in pairs[:top_k]]
            except Exception:
                results = []

    try:
        if client is not None:
            client.close()
    except Exception:
        pass
    return results
