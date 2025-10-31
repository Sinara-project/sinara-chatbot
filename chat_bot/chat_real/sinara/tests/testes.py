import http.client
import json
import logging
from time import sleep
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional

# Configuration
CONFIG = {
    "url": "127.0.0.1",
    "port": 8200,
    "endpoint": "/api/chat",
    "max_retries": 20,
    "timeout": 15
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRunner:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        # Alinha para usar a pasta 'data' existente
        self.questions_dir = self.base_dir / "data"
        self.answers_dir = self.base_dir / "data"
        self.answers_dir.mkdir(parents=True, exist_ok=True)
        self.test_types = ("default", "excecao", "alucinacao", "memoria")

    def run_tests(self):
        for test_type in self.test_types:
            questions = self._load_questions(test_type)
            answers = self._process_questions(questions)
            self._save_answers(test_type, answers)
            
    def _load_questions(self, test_type: str) -> List[Dict]:
        q_path = self.questions_dir / f"{test_type}_pergunta.json"
        if not q_path.exists():
            raise FileNotFoundError(f"Question file not found: {q_path}")
        
        with q_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _process_questions(self, questions: List[Dict]) -> List[Dict]:
        answers = []
        for item in questions:
            payload = {
                "query": item["message"],
                "session_id": str(item.get("session_id", "")),
                "agent": "auto",
            }
            
            url_final = CONFIG['endpoint']
            response = self._make_request(url_final, method="POST", body=json.dumps(payload))
            
            answers.append({
                "session_id": item.get("session_id"),
                "pergunta": item["message"],
                "resposta": response
            })
            sleep(2)
        return answers

    def _make_request(self, url: str, method: str = "GET", body: Optional[str] = None, headers: Optional[Dict] = None) -> str:
        """
        Realiza requisição HTTP com retry. Suporta GET e POST.
        - url: caminho (ex: /api/chat)
        - method: "GET" ou "POST"
        - body: string (JSON) para POST
        - headers: dict de cabeçalhos adicionais
        """
        for attempt in range(CONFIG["max_retries"]):
            conn = http.client.HTTPConnection(
                CONFIG["url"], 
                CONFIG["port"],
                timeout=CONFIG["timeout"]
            )
            try:
                req_headers = {"Accept": "application/json", "Connection": "close"}
                if headers:
                    req_headers.update(headers)
                if method.upper() == "POST":
                    req_headers.setdefault("Content-Type", "application/json")
                    conn.request("POST", url, body=body or "", headers=req_headers)
                else:
                    # URL já deve incluir query string se necessário
                    conn.request("GET", url, headers=req_headers)
                response = conn.getresponse()
                response_body = response.read().decode("utf-8", errors="ignore")
                
                if response.status == 200:
                    return self._parse_response(response_body)
                else:
                    logger.error(f"HTTP {response.status}: {response.reason} - tentativa {attempt+1}")
                    
            except Exception as e:
                logger.error(f"Request failed (tentativa {attempt+1}): {e}")
                
            finally:
                try:
                    conn.close()
                except:
                    pass
                
            sleep(2)
        return "Error: Max retries exceeded"

    def _parse_response(self, response_body: str) -> str:
        try:
            payload = json.loads(response_body)
            # tenta várias chaves possíveis
            return payload.get("answer") or payload.get("resposta") or payload.get("data") or "Response error"
        except json.JSONDecodeError:
            return self._handle_invalid_json(response_body)

    def _handle_invalid_json(self, response_body: str) -> str:
        start = response_body.find("{")
        end = response_body.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(response_body[start:end + 1])
                return payload.get("answer") or payload.get("resposta") or "Response error"
            except:
                pass
        return "JSON decode error"

    def _save_answers(self, test_type: str, answers: List[Dict]):
        a_path = self.answers_dir / f"{test_type}_resposta.json"
        with a_path.open("w", encoding="utf-8") as f:
            json.dump(answers, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved responses to {a_path.name}")

if __name__ == "__main__":
    runner = TestRunner()
    runner.run_tests()
