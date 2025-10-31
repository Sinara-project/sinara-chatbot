import os
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .rag_agent_assistente import run_rag_agent_assistente
from .rag_agent_tecnico import run_rag_agent_tecnico
from .rag_agent_organizacional import run_rag_agent_organizacional
from .faq_agent import run_faq_agent
from ..services.rag_service import retrieve_similar_context_with_scores


load_dotenv(override=True)
TZ = ZoneInfo("America/Sao_Paulo")
today = datetime.now(TZ).date()


def _llm(model: str | None = None, temperature: float = 0.0):
    api = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api:
        return None
    m = model or os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash-latest")
    return ChatGoogleGenerativeAI(model=m, google_api_key=api, temperature=temperature)


# ----------------- Roteador -----------------
_system_router = (
    "system",
    f"""
### PERSONA SISTEMA
Você é o Roteador do Sinara. É objetivo, responsável e confiável.
- Respostas curtas e aplicáveis.
- Hoje é {today.isoformat()} (America/Sao_Paulo).

### PAPEL
- Decidir a rota: (assistente | tecnico | organizacional | faq).
- Responder diretamente em: (a) saudações/small talk, (b) fora de escopo (redirecionar).
- Quando for caso de especialista, NÃO responda; apenas encaminhe a mensagem ORIGINAL e a PERSONA.

### REGRAS
- Seja breve e objetivo.
- Se faltar um dado essencial para decidir a rota, faça UMA pergunta (CLARIFY), senão deixe vazio.
- FAQ: dúvidas sobre regras/funcionalidades documentadas (p.ex. bater ponto, como usar o app).
- Tecnico: código/infra/API/erros.
- Organizacional: políticas/processos internos.
- Assistente: dúvidas gerais de uso que não sejam exclusivamente FAQ.

### PROTOCOLO (texto puro)
ROUTE=<assistente|tecnico|organizacional|faq>
PERGUNTA_ORIGINAL=<mensagem completa do usuário>
PERSONA=<copie PERSONA SISTEMA>
CLARIFY=<pergunta mínima ou vazio>
""",
)

_example_router = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}"),
])

_shots_router = FewShotChatMessagePromptTemplate(
    examples=[
        {"human": "Oi, tudo bem?", "ai": "Olá! Posso te ajudar com o sistema; o que você precisa?"},
        {"human": "Como bater ponto?", "ai": "ROUTE=faq\nPERGUNTA_ORIGINAL=Como bater ponto?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="},
        {"human": "Erro 500 no endpoint /chat", "ai": "ROUTE=tecnico\nPERGUNTA_ORIGINAL=Erro 500 no endpoint /chat\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="},
        {"human": "Quais são as regras de férias?", "ai": "ROUTE=organizacional\nPERGUNTA_ORIGINAL=Quais são as regras de férias?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="},
        {"human": "Como usar o formulário de ocorrência?", "ai": "ROUTE=assistente\nPERGUNTA_ORIGINAL=Como usar o formulário de ocorrência?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="},
    ],
    example_prompt=_example_router,
)

_router_prompt = ChatPromptTemplate.from_messages([
    _system_router,
    _shots_router,
    ("human", "{input}"),
])

def _get_router_chain():
    llm = _llm(temperature=0)
    if llm is None:
        return None
    return _router_prompt | llm | StrOutputParser()


# ----------------- Orquestrador -----------------
_system_orchestrator = (
    "system",
    """
Você é o Orquestrador. Formate a resposta final ao usuário com base no JSON do especialista.

REGRAS
- Use exatamente o campo 'resposta' como primeira linha.
- Se 'recomendacao' existir e não for vazia, inclua seção "- Recomendação:".
- Para acompanhamento: se houver 'esclarecer', use-o; senão, se houver 'acompanhamento', use-o; caso contrário omita.
- Não retorne JSON, apenas texto final conciso.
""",
)

_orchestrator_prompt = ChatPromptTemplate.from_messages([
    _system_orchestrator,
    ("human", "ESPECIALISTA_JSON:\n{json_text}"),
])

def _get_orchestrator_chain():
    llm = _llm(temperature=0)
    if llm is None:
        return None
    return _orchestrator_prompt | llm | StrOutputParser()


def _wrap_json(dominio: str, resposta: str, recomendacao: str = "", esclarecer: str = "", acompanhamento: str = "") -> str:
    # JSON textual para o orquestrador
    parts: Dict[str, Any] = {
        "dominio": dominio,
        "intencao": "responder",
        "resposta": resposta.strip(),
    }
    if recomendacao:
        parts["recomendacao"] = recomendacao
    if esclarecer:
        parts["esclarecer"] = esclarecer
    if acompanhamento:
        parts["acompanhamento"] = acompanhamento
    # JSON manual simples (sem depender de dumps por consistência com few-shots)
    items = []
    for k, v in parts.items():
        v_str = str(v).replace("\n", " ")
        items.append(f'"{k}":"{v_str}"')
    return "{" + ",".join(items) + "}"


def executar_fluxo(query: str, session_id: str) -> str:
    # 0) Heurística early para priorizar técnico/organizacional e evitar quedas no assistente
    q = (query or "").strip()
    qlow = q.lower()
    tecnico_kw = [
    # Parâmetros de qualidade
    "ph", "turbidez", "ntu", "cor aparente", "cor verdadeira", "tds", "std", "sdt",
    "sólidos totais", "sólidos suspensos", "condutividade", "temperatura", "orp",
    "alcalinidade", "dureza", "ferro", "manganês", "amônia", "nitrato", "nitrito",
    "fluoreto", "cloro livre", "cloro total", "cloro residual", "cto", "trihalometanos",
    "haas", "odor", "sabor", "geosmina", "mib", "coliformes", "e. coli", "heterotróficos",

    # Processos/etapas
    "pré-oxidação", "pré-cloração", "mistura rápida", "coagulação", "floculação",
    "decantação", "clarificação", "lamelar", "flotação", "filtração", "carreira de filtração",
    "perda de carga", "retrolavagem", "backwash", "expansão de leito", "ar-água",
    "desinfecção", "pós-cloração", "fluoretação", "tempo de contato", "ct",

    # Produtos químicos
    "hipoclorito", "cloro gás", "dióxido de cloro", "permanganato", "sulfato de alumínio",
    "pacs", "pac", "cloreto férrico", "sulfato férrico", "cal hidratada", "cal virgem",
    "soda cáustica", "ácido sulfúrico", "carvão ativado em pó", "cap", "carvão ativado granular", "cag",
    "polímero", "polímero catiônico", "polímero aniônico", "antiespumante",

    # Operação & controle
    "dosagem", "dosar", "ponto de dosagem", "demanda de cloro", "jar test", "teste de jarros",
    "potencial zeta", "g", "gt", "ph de coagulação", "correção de alcalinidade",
    "setpoint", "pid", "scada", "clp", "intertravamento", "alarmes",

    # Equipamentos
    "bomba dosadora", "dosadora diafragma", "dosadora peristáltica", "dosadora pistão",
    "misturador estático", "medidor de vazão", "rotâmetro", "válvula gaveta", "válvula borboleta",
    "turbidímetro", "phmetro", "condutivímetro", "clorímetro", "colorímetro", "espectrofotômetro",
    "oxirredução", "coagulador mecânico", "floculador", "decantador", "filtro rápido de areia",
    "filtro dual media", "filtro antracito", "unidade de retrolavagem", "ejetor de cloro",

    # Hidráulica & projeto operacional
    "tas", "taxa de aplicação superficial", "taxa de filtração", "m3/m2.h", "carga hidráulica",
    "npsh", "cavitação", "curva q-h", "perda de carga localizada", "coeficiente k",

    # Lodo & resíduos
    "lodo de ETA", "adensamento", "leito de secagem", "polímero para lodo", "desaguamento",

    # Amostragem & laboratório
    "plano de amostragem", "ponto de coleta", "cadeia de custódia",
    "incerteza", "calibração", "branco", "padrão", "curva de calibração",

    # Conformidade & referência (palavras de intenção)
    "portaria 888", "potabilidade", "conformidade", "parâmetro fora do padrão",
    "plano de segurança da água", "psa", "registro de operação", "boletim diário"
]
    
    organizacional_kw = [
    # Governança & políticas
    "política", "políticas internas", "normas internas", "compliance",
    "governança", "código de conduta", "lgpd", "sigilo", "confidencialidade",

    # Processos & procedimentos
    "processo", "processos", "procedimento", "procedimentos",
    "pop", "procedimento operacional padrão", "instrução de trabalho",
    "fluxo de trabalho", "workflow", "mapeamento de processos",
    "padronização", "melhoria contínua", "kaizen", "5s",

    # Documentação & controle
    "documentação", "documentos", "versão de documento", "controle de versão",
    "template", "modelo de documento", "checklist", "registro", "histórico",
    "auditoria de documentos", "trilha de auditoria", "rastreabilidade",

    # RH & pessoas
    "rh", "recursos humanos", "onboarding", "integração",
    "offboarding", "desligamento", "cargo", "função", "papel",
    "perfis de acesso", "permissões", "escala", "turno", "alocação",
    "treinamento", "capacitação", "matriz de competências",
    "avaliação de desempenho", "feedback", "ead",

    # Pontos & jornada
    "bater ponto", "registro de ponto", "apontamento de horas",
    "jornada de trabalho", "frequência", "assiduidade",

    # Segurança, saúde e meio ambiente (SSMA)
    "sst", "ssma", "segurança do trabalho", "epi", "epc",
    "nr", "nr-12", "nr-33", "nr-35", "permite de trabalho", "pt",
    "análise de risco", "perigo", "risco", "mitigação",
    "meio ambiente", "licença ambiental", "resíduos", "evidência de conformidade",

    # Qualidade & auditoria
    "qualidade", "iso", "iso 9001", "iso 14001", "auditoria",
    "não conformidade", "nc", "ação corretiva", "ação preventiva",
    "plano de ação", "5 porquês", "ishikawa",

    # Planejamento & indicadores
    "planejamento", "cronograma", "sprint", "roadmap",
    "kpi", "indicadores", "metas", "sla", "okr", "dashboard gerencial",

    # Comunicação & ritos
    "reunião", "ata", "pauta", "comunicado", "memorando",
    "dss", "dds", "briefing", "alinhamento", "handover", "passagem de turno",

    # Metodologias ágeis
    "metodologia ágil", "agile", "scrum", "kanban", "daily",
    "retrospectiva", "planning", "backlog", "product owner", "scrum master",

    # Organização & estrutura
    "organização", "organograma", "estrutura organizacional",
    "matriz raci", "responsáveis", "governança de dados",

    # Conformidade legal e regulatória (ETA/indústria)
    "licenças", "regulatório", "conformidade legal",
    "procedimentos eta", "padrões operacionais", "inspeção", "auditoria interna",

    # Relatórios & aprovações
    "relatório gerencial", "relatório operacional", "homologação",
    "aprovação", "validação", "liberação"
]

    if any(k in qlow for k in tecnico_kw):
        route = "tecnico"
    elif any(k in qlow for k in organizacional_kw):
        route = "organizacional"
    else:
        # 0.1) Sinal forte de FAQ pelo contexto.json
        route = None
        try:
            pairs = retrieve_similar_context_with_scores(q, top_k=3)
            top_score = pairs[0][0] if pairs else 0.0
            if top_score >= 0.65:
                route = "faq"
        except Exception:
            pass

    if not route:
        # 1) Roteamento por LLM (pode retornar resposta direta sem ROUTE=)
        chain = _get_router_chain()
        if chain is not None:
            try:
                routed = chain.invoke({"input": q})
                if "ROUTE=" not in (routed or ""):
                    return routed
                # 2) Direciona conforme protocolo
                route = "assistente"
                for key in ("assistente", "tecnico", "organizacional", "faq"):
                    if f"ROUTE={key}" in routed:
                        route = key
                        break
            except Exception:
                pass

    if route == "faq":
        resposta, contexto = run_faq_agent(query)
        json_text = _wrap_json("faq", resposta)
        orch = _get_orchestrator_chain()
        if orch is not None:
            try:
                final = orch.invoke({"json_text": json_text})
                return final
            except Exception:
                pass
        return resposta

    # Usa os agentes existentes para gerar resposta e envolve em JSON simples
    if route == "assistente":
        resposta, contexto = run_rag_agent_assistente(query, session_id)
        json_text = _wrap_json("assistente", resposta)
    elif route == "tecnico":
        resposta, contexto = run_rag_agent_tecnico(query, session_id)
        json_text = _wrap_json("tecnico", resposta)
    else:  # organizacional
        resposta, contexto = run_rag_agent_organizacional(query, session_id)
        json_text = _wrap_json("organizacional", resposta)

    orch = _get_orchestrator_chain()
    if orch is not None:
        try:
            final = orch.invoke({"json_text": json_text})
            return final
        except Exception:
            pass
    return resposta
