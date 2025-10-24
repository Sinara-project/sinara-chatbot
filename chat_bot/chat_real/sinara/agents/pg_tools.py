# Ferramentas (tools) para agentes do app operacional: Assistente, Técnico e Organizacional.

import os, json
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import DictCursor
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo
from datetime import datetime
from pydantic import BaseModel, Field, validator
from langchain.tools import tool
from dotenv import load_dotenv

# Carrega variáveis do .env para o processo ANTES de instanciar o LLM
load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Defina GEMINI_API_KEY (ou GOOGLE_API_KEY) no ambiente/.env antes de iniciar."
    )


load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
TZ = ZoneInfo("America/Sao_Paulo")

def get_conn():
    return psycopg2.connect(DATABASE_URL)

# ---------------------------
# Schemas Pydantic
# ---------------------------

class CriarForms(BaseModel):
    name: str
    schema_json: Dict[str, Any]
    version: Optional[str] = "1.0"
    is_active: bool = True

class ListarFormsArgs(BaseModel):
    active_only: bool = True
    q: Optional[str] = None
    limit: int = 50

class PreencherForms(BaseModel):
    form_id: int
    operator_id: Optional[int] = None
    occurred_at: Optional[str] = None
    data: Dict[str, Any]

    @validator("occurred_at")
    def _validate_iso(cls, v):
        if v is None:
            return v
        try:
            datetime.fromisoformat(v.replace("Z","+00:00"))
            return v
        except Exception:
            raise ValueError("occurred_at deve estar em ISO 8601")

class ConsultarForms(BaseModel):
    form_id: int
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    limit: int = 200

class AlertRulesArgs(BaseModel):
    form_id: int
    date_local: str
    rules: Dict[str, Dict[str, Optional[float]]]

class FaqSearchArgs(BaseModel):
    query: str
    limit: int = 5

class ShiftScheduleArgs(BaseModel):
    date_local: str
    team: Optional[str] = None
    limit: int = 20

# ---------------------------
# Funções auxiliares
# ---------------------------

def _optional_date_clause(field: str, df: Optional[str], dt: Optional[str], args: list) -> str:
    clauses = []
    if df and dt:
        clauses.append(f"{field}::date BETWEEN %s::date AND %s::date")
        args.extend([df, dt])
    elif df:
        clauses.append(f"{field}::date >= %s::date")
        args.append(df)
    elif dt:
        clauses.append(f"{field}::date <= %s::date")
        args.append(dt)
    return " AND ".join(clauses)

def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

# ---------------------------
# TOOLS
# ---------------------------

# Ferramentas do agente Assistente (gestão de formulários, entradas, alertas, FAQs, escala)

@tool("criar_form", args_schema=CriarForms)
def criar_form(name: str, schema_json: Dict[str, Any], version: Optional[str]="1.0", is_active: bool=True) -> dict:
    """Cria ou atualiza um formulário."""
    conn = get_conn(); cur = conn.cursor(cursor_factory=DictCursor)
    try:
        cur.execute("""
            INSERT INTO forms (name, schema, version, is_active, created_at)
            VALUES (%s, %s::jsonb, %s, %s, NOW())
            ON CONFLICT (name) DO UPDATE SET schema = EXCLUDED.schema, version = EXCLUDED.version, is_active = EXCLUDED.is_active
            RETURNING id, name, version, is_active, created_at;
        """, (name, json.dumps(schema_json), version, is_active))
        row = cur.fetchone(); conn.commit()
        return {"status":"ok", "form": dict(row)}
    except Exception as e:
        conn.rollback(); return {"status":"error","message":str(e)}
    finally:
        cur.close(); conn.close()

@tool("list_forms", args_schema=ListarFormsArgs)
def list_forms(active_only: bool=True, q: Optional[str]=None, limit: int=50) -> dict:
    """Lista formulários disponíveis."""
    conn = get_conn(); cur = conn.cursor(cursor_factory=DictCursor)
    try:
        sql = "SELECT id, name, version, is_active, created_at FROM forms"
        args = []
        wh = []
        if active_only:
            wh.append("is_active = TRUE")
        if q:
            wh.append("name ILIKE %s")
            args.append(f"%{q}%")
        if wh:
            sql += " WHERE " + " AND ".join(wh)
        sql += " ORDER BY name ASC LIMIT %s"
        args.append(int(limit))
        cur.execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()]
        return {"status":"ok","forms":rows}
    except Exception as e:
        return {"status":"error","message":str(e)}
    finally:
        cur.close(); conn.close()

@tool("submit_form", args_schema=PreencherForms)
def submit_form(form_id: int, data: Dict[str, Any], operator_id: Optional[int]=None, occurred_at: Optional[str]=None) -> dict:
    """Envia uma entrada de formulário."""
    conn = get_conn(); cur = conn.cursor(cursor_factory=DictCursor)
    try:
        if occurred_at:
            cur.execute("""
                INSERT INTO form_entries(form_id, data, operator_id, occurred_at, created_at)
                VALUES (%s, %s::jsonb, %s, %s::timestamptz, NOW())
                RETURNING id, form_id, occurred_at, operator_id, created_at;
            """, (form_id, json.dumps(data), operator_id, occurred_at))
        else:
            cur.execute("""
                INSERT INTO form_entries(form_id, data, operator_id, occurred_at, created_at)
                VALUES (%s, %s::jsonb, %s, NOW(), NOW())
                RETURNING id, form_id, occurred_at, operator_id, created_at;
            """, (form_id, json.dumps(data), operator_id))
        row = cur.fetchone(); conn.commit()
        return {"status":"ok","entry":dict(row)}
    except Exception as e:
        conn.rollback(); return {"status":"error","message":str(e)}
    finally:
        cur.close(); conn.close()

@tool("list_entries", args_schema=ConsultarForms)
def list_entries(form_id: int, date_from: Optional[str]=None, date_to: Optional[str]=None, limit: int=200) -> dict:
    """Lista entradas de um formulário."""
    conn = get_conn(); cur = conn.cursor(cursor_factory=DictCursor)
    try:
        sql = "SELECT id, form_id, data, occurred_at, operator_id, created_at FROM form_entries WHERE form_id = %s"
        args = [form_id]
        clause = _optional_date_clause("occurred_at", date_from, date_to, args)
        if clause:
            sql += " AND " + clause
        sql += " ORDER BY occurred_at DESC LIMIT %s"; args.append(int(limit))
        cur.execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()]
        return {"status":"ok","entries":rows}
    except Exception as e:
        return {"status":"error","message":str(e)}
    finally:
        cur.close(); conn.close()

@tool("alert_if_out_of_spec", args_schema=AlertRulesArgs)
def alert_if_out_of_spec(form_id: int, date_local: str, rules: Dict[str, Dict[str, Optional[float]]]) -> dict:
    """Verifica leituras fora dos limites e retorna alertas."""
    conn = get_conn(); cur = conn.cursor(cursor_factory=DictCursor)
    try:
        cur.execute("""
            SELECT id, data, occurred_at, operator_id
            FROM form_entries
            WHERE form_id = %s AND occurred_at::date = %s::date
            ORDER BY occurred_at ASC
        """, (form_id, date_local))
        alerts = []
        for r in cur.fetchall():
            payload = r["data"] or {}
            for key, rule in rules.items():
                val = _safe_float(payload.get(key))
                if val is None: 
                    continue
                lo = rule.get("min"); hi = rule.get("max")
                out = (lo is not None and val < lo) or (hi is not None and val > hi)
                if out:
                    alerts.append({
                        "entry_id": r["id"],
                        "occurred_at": str(r["occurred_at"]),
                        "metric": key,
                        "value": val,
                        "limits": rule,
                        "operator_id": r["operator_id"]
                    })
        return {"status":"ok","date":date_local,"form_id":form_id,"alerts":alerts, "count": len(alerts)}
    except Exception as e:
        return {"status":"error","message":str(e)}
    finally:
        cur.close(); conn.close()

@tool("faq_search", args_schema=FaqSearchArgs)
def faq_search(query: str, limit: int=5) -> dict:
    """Busca respostas rápidas em FAQs."""
    conn = get_conn(); cur = conn.cursor(cursor_factory=DictCursor)
    try:
        cur.execute("""
            SELECT id, question, answer, audience
            FROM faqs
            WHERE question ILIKE %s OR answer ILIKE %s
            ORDER BY id DESC
            LIMIT %s
        """, (f"%{query}%", f"%{query}%", int(limit)))
        rows = [dict(r) for r in cur.fetchall()]
        return {"status":"ok","results":rows}
    except Exception as e:
        return {"status":"error","message":str(e)}
    finally:
        cur.close(); conn.close()

@tool("get_shift_schedule", args_schema=ShiftScheduleArgs)
def get_shift_schedule(date_local: str, team: Optional[str]=None, limit: int=20) -> dict:
    """Consulta escala de turnos."""
    conn = get_conn(); cur = conn.cursor(cursor_factory=DictCursor)
    try:
        sql = """
            SELECT id, date_local, team, operator_name, start_time, end_time, notes
            FROM shifts WHERE date_local = %s
        """
        args = [date_local]
        if team:
            sql += " AND team = %s"
            args.append(team)
        sql += " ORDER BY start_time ASC LIMIT %s"
        args.append(int(limit))
        cur.execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()]
        return {"status":"ok","date":date_local,"shifts":rows}
    except Exception as e:
        return {"status":"error","message":str(e)}
    finally:
        cur.close(); conn.close()

# ---------------------------
# Listas de tools por agente
# ---------------------------

# Agente Assistente: gestão completa
ETA_TOOLS_ASSISTENTE = [
    criar_form, list_forms, submit_form
 
]

# Agente Técnico: consulta e alertas
ETA_TOOLS_TECNICO = [
    list_entries, alert_if_out_of_spec
]

# Agente Organizacional: FAQs e escala
ETA_TOOLS_ORGANIZACIONAL = [
    faq_search, get_shift_schedule
]
