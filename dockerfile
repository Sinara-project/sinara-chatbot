FROM python:3.11-slim
WORKDIR /app

# Copy the actual requirements file from the project
COPY chat_bot/chat_real/sinara/requirements.txt /app/requirements.txt

# Install dependencies (requirements already pin LangChain and friends)
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY chat_bot /app/chat_bot
ENV PYTHONPATH=/app PORT=8080
CMD uvicorn chat_bot.chat_real.sinara.main:app --host 0.0.0.0 --port 8080
