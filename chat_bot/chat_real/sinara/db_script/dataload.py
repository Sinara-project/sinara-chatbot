import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def main():
    load_dotenv(override=True)

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DB", "DB_Sinara")
    if not mongo_uri:
        raise RuntimeError("Defina MONGO_URI no .env/ambiente.")

    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    collection = db["contexto"]

    # Caminho do JSON relativo a este arquivo
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, "contexto.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        contexto_docs = json.load(f)

    if not isinstance(contexto_docs, list):
        raise ValueError("contexto.json deve conter uma lista de documentos")

    # Gera embeddings para registros que não possuem 'embedding'
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Defina GEMINI_API_KEY (ou GOOGLE_API_KEY) no .env/ambiente.")

    emb = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
        transport="rest",
    )

    enriched = []
    for doc in contexto_docs:
        content = doc.get("content") or doc.get("conteudo")
        if content and not doc.get("embedding"):
            try:
                vec = emb.embed_query(content)
                doc["embedding"] = vec
            except Exception as e:
                print(f"Falha ao gerar embedding para título '{doc.get('title') or ''}': {e}")
        enriched.append(doc)

    # Substitui a coleção
    collection.delete_many({})
    if enriched:
        collection.insert_many(enriched)
    client.close()

    print(f"{len(enriched)} documentos inseridos na coleção 'contexto'.")


if __name__ == "__main__":
    main()
