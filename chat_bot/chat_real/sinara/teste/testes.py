import http.client
import json
from time import sleep
import urllib.parse

# Configurações da API do chat-real (Sinara)
url = "127.0.0.1"
port = 8200
endpoint = "/chat"
headers = {
    "Content-Type": "application/json"
}

# Tipos de teste e caminhos corretos para o chat-real
test_types = ('default', 'excecao', 'alucinacao', 'memoria')

for test_type in test_types:
    # Lê as perguntas do arquivo JSON correto
    with open(f'chat_real/sinara/teste/questions/{test_type}_pergunta.json', 'r', encoding='utf-8') as file:
        questions = json.load(file)

    answers = []

    # Conecta com o servidor
    conn = http.client.HTTPConnection(url, port)

    for item in questions:
        # Monta a query string com os parâmetros
        params = urllib.parse.urlencode({
            "query": item["message"],
            "session_id": str(item["session_id"])
        })

        # URL final com query string
        url_final = f"{endpoint}?{params}"
        print(url_final)

        # Fazendo tentativas de requisição, caso o status não seja 200
        for i in range(20):
            conn.request("GET", url_final, headers=headers)
            response = conn.getresponse()

            if response.status == 200:
                break
            sleep(5)

        # Lê a resposta
        response_body = response.read().decode("utf-8")
        try:
            reply_message = json.loads(response_body).get("resposta", "Erro na resposta")
        except json.JSONDecodeError:
            reply_message = "Erro ao decodificar JSON"

        answers.append({
            "session_id": item["session_id"],
            "pergunta": item["message"],
            "resposta": reply_message
        })

    # Fecha a conexão
    conn.close()

    # Grava as respostas no arquivo correto
    with open(f"chat_real/sinara/teste/answers/{test_type}_resposta.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)

    print(f"Todas as perguntas foram enviadas e respostas salvas em '{test_type}_resposta.json'.")

    sleep(10)