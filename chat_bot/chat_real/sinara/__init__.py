"""Sinara chatbot package.

Ensures environment variables from this package's .env are loaded
even when the application is launched from a different working
directory. This avoids missing API keys (e.g., GEMINI_API_KEY)
that cause agents to fall back with generic error messages.
"""

from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv


# Load .env that sits alongside this package, independent of CWD
_ENV_PATH = Path(__file__).resolve().with_name(".env")
try:
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
except Exception:
    # If anything goes wrong, don't crash import; modules also attempt
    # loading .env individually. This just improves reliability.
    pass

