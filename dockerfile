FROM python:3.11-slim
WORKDIR /app
COPY chat_bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       langchain==0.1.16 \
       langchain-core==0.1.52 \
       langchain-google-genai==0.0.9 \
       langchain-mongodb==0.1.1 \
       pymongo>=4.6.0 \
       numpy>=1.26.0
COPY chat_bot /app/chat_bot
ENV PYTHONPATH=/app PORT=8080
CMD uvicorn chat_bot.chat_real.sinara.main:app --host 0.0.0.0 --port 8080
