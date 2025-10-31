"""
Módulo de testes do Chatbot Sinara
"""

import os

# Define caminho base para dados de teste
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Verifica se diretório existe
if not os.path.exists(TEST_DATA_DIR):
    os.makedirs(TEST_DATA_DIR)