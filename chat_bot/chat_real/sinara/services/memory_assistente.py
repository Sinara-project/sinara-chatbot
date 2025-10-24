from langchain_mongodb import MongoDBChatMessageHistory
import os
from dotenv import load_dotenv

load_dotenv(override=True)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "DB_Sinara")

def get_memory(session_id: str):
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_URI,
        database_name=MONGO_DB,
        collection_name="conversation_assistente",
    )
