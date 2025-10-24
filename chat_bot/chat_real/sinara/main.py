from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Load .env colocated with this module, fallback to default search
_dotenv_loaded = load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
if not _dotenv_loaded:
    load_dotenv(override=True)

from .pipeline import run_pipeline
from .services.memory_assistente import get_memory as get_memory_assistente
from .services.memory_tecnico import get_memory as get_memory_tecnico
from .services.rag_service import retrieve_similar_context

app = FastAPI(title="API do Chatbot Sinara")

# CORS: libera chamadas do browser/Swagger e apps locais
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API do Chatbot Sinara está online!"}

class ChatBody(BaseModel):
    query: str
    session_id: Optional[str] = None
    agent: str = "assistente"

def _validate_agent(agent: str, session_id: Optional[str]) -> Optional[str]:
    if agent not in ("assistente", "tecnico", "organizacional"):
        return "O parâmetro 'agent' deve ser 'assistente', 'tecnico' ou 'organizacional'."
    return None

@app.get("/chat")
def chat(query: Optional[str] = None, session_id: Optional[str] = None, agent: str = "assistente"):
    if not query:
        return {"ok": False, "error": "query vazia"}

    err = _validate_agent(agent, session_id)
    if err:
        return {"ok": False, "error": err}

    # Opcional: recuperar histórico/contexto local se houver session_id (pipeline também gerencia)
    history = []
    if session_id:
        if agent == "assistente":
            memory = get_memory_assistente(session_id)
            history = getattr(memory, "messages", [])
        elif agent == "tecnico":
            memory = get_memory_tecnico(session_id)
            history = getattr(memory, "messages", [])

    contextos = retrieve_similar_context(query, top_k=3)

    # Chama pipeline principal (faz a geração real)
    answer = run_pipeline(query=query, session_id=session_id, agent=agent)

    return {
        "ok": True,
        "agent": agent,
        "session_id": session_id,
        "history": history,
        "contexts": contextos,
        "answer": answer,
    }

@app.post("/chat")
def chat_post(body: ChatBody):
    err = _validate_agent(body.agent, body.session_id)
    if err:
        return {"ok": False, "error": err}

    answer = run_pipeline(query=body.query, session_id=body.session_id, agent=body.agent)
    return {"ok": True, "agent": body.agent, "session_id": body.session_id, "answer": answer}

@app.get("/health")
def health():
    return {"status": "ok"}
