```mermaid
flowchart TD
  U[Usuário] --> API[FastAPI /api/chat]
  API --> ENV[.env / Settings]
  API --> RAG[Retrieve Contexts\n(contexto.json → Embeddings Gemini ou BM25)]
  RAG --> ROUTER[Router Agent\n(assistente | tecnico | organizacional | faq)]

  %% Caminho Assistente (com Guardrail/Judge)
  ROUTER -- assistente --> G[Guardrail Agent]
  G --> ASSIST[Agente Assistente\n+ possível desvio p/ Organizacional baseado no contexto]
  ASSIST --> LLM1[LLM (Gemini)\nvia LangChain]
  LLM1 --> J[Judge Agent]
  J --> RESP1[Resposta API → Frontend\n+ grava no histórico]

  %% Caminho Técnico (direto no código atual)
  ROUTER -- tecnico --> TECH[RAG Agente Técnico]
  TECH --> LLM2[LLM (Gemini)]
  LLM2 --> RESP2[Resposta API → Frontend]

  %% Caminho Organizacional (quando roteado diretamente)
  ROUTER -- organizacional --> ORG[RAG Agente Organizacional]
  ORG --> LLM3[LLM (Gemini)]
  LLM3 --> RESP3[Resposta API → Frontend]

  %% FAQ
  ROUTER -- faq --> FAQ[FAQ Agent]
  FAQ --> RESP4[Resposta API → Frontend]

  %% Memória (usada principalmente no fluxo assistente)
  subgraph Persistence
    M[(MongoDB Chat History)]
    C[(contexto.json)]
  end
  U -->|salva user msg| M
  J -->|salva resposta| M
  C --- RAG


```