import os
from dotenv import load_dotenv
from langchain_mongodb import MongoDBChatMessageHistory

load_dotenv(override=True)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "DB_Sinara")

# Coleção para salvar o histórico 
COLLECTION = "chat_history"

def get_memory(session_id: str):
    """
    Retorna um MongoDBChatMessageHistory para a sessão informada.
    Cada sessão fica registrada em um documento com o id = session_id.
    """
    return MongoDBChatMessageHistory(
        connection_string=MONGO_URI,
        database_name=MONGO_DB,
        collection_name=COLLECTION,
        session_id=session_id,
    )

