import os
import psycopg
from pathlib import Path
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")
def get_connection():
    return psycopg.connect(
        host=os.getenv("VEYRA_DB_HOST", "127.0.0.1"),
        port=int(os.getenv("VEYRA_DB_PORT", "5432")),
        dbname=os.getenv("VEYRA_DB_NAME", "veyra"),
        user=os.getenv("VEYRA_DB_USER", "veyra"),
        password=os.environ["VEYRA_DB_PASSWORD"],
        connect_timeout=5,
    )