# Chatbot Sinara

Sistema de chatbot para suporte em ETAs (Estações de Tratamento de Água)

## Estrutura do Projeto

```
chat_bot/
└─ chat_real/
   └─ sinara/
      ├─ agents/     # Agentes de processamento
      ├─ api/        # Rotas e modelos da API
      ├─ config/     # Configurações
      ├─ db_script/  # Scripts de banco de dados
      ├─ logs/       # Arquivos de log
      ├─ prompts/    # Templates de prompts
      ├─ services/   # Serviços compartilhados
      ├─ tests/      # Testes automatizados
      └─ utils/      # Utilitários
```

## Instalação

1. Clone o repositório
2. Crie um ambiente virtual: `python -m venv venv`
3. Ative o ambiente: `.\\venv\\Scripts\\activate`
4. Instale as dependências: `pip install -r requirements.txt`
5. Configure o arquivo .env baseado no .env.example
6. Execute: `uvicorn chat_real.sinara.main:app --reload`

