# Chat Real — Ajustes inspirados no `embedding`

Este pacote integra à API do Sinara (FastAPI) o pipeline completo de guardrail → RAG → juiz,
mantendo a estrutura original e adicionando apenas o necessário.

## O que foi feito
- **`/sinara/main.py`**: agora usa `run_pipeline(...)` tanto para `GET /chat` quanto para `POST /chat` (JSON).
- **Compatibilidade de nomes**: adicionados alias `agents/judge_agent.py` e `agents/pg_tools.py`, apontando para os arquivos existentes
  (`jugde_agent.py` e `pg_tolls.py`), evitando mudanças de import.
- **`requirements.txt` (raiz do projeto)**: lista dependências mínimas para rodar localmente.
- **`.env.example`**: modelo de variáveis de ambiente.
- **`/health`**: endpoint simples de saúde da API.

## Como rodar
1. Crie o arquivo de variáveis:
   ```bash
   cp chat_real/sinara/.env.example chat_real/sinara/.env
   # edite GOOGLE_API_KEY, MONGO_URI, MONGO_DB
   ```
2. Instale dependências (idealmente em um venv):
   ```bash
   pip install -r requirements.txt
   ```
3. Suba a API (porta 8200 mantida como no projeto):
   ```bash
   cd chat_real/sinara
   uvicorn main:app --host 0.0.0.0 --port 8200 --reload
   ```

## Como usar
- **GET**:
  ```
  /chat?query=Olá&session_id=123&agent=assistente
  ```
- **POST**:
  ```json
  POST /chat
  {
    "query": "Olá",
    "session_id": "123",
    "agent": "assistente"
  }
  ```

## Observações
- Mantivemos a estrutura de pastas original. Nada foi removido.
- O `pipeline.py` já orquestra Guardrail → RAG → Judge como no `embedding`.
- Caso tenha modelos/rotas extras do `embedding` que deseje replicar (ex.: endpoints de _programs_ ou _search_ dedicados),
  dá para adicionar facilmente em `main.py` seguindo o mesmo padrão.
