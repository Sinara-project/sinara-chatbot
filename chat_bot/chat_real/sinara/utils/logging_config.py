import logging
import os
from datetime import datetime

def setup_logging():
    """Configura sistema de logs"""
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    formato = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    arquivo = os.path.join(logs_dir, f"sinara_{datetime.now():%Y%m%d}.log")

    logging.basicConfig(
        level=logging.INFO,
        format=formato,
        handlers=[
            logging.FileHandler(arquivo, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )