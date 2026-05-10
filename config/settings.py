from pathlib import Path
import os
import tomllib

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


def _load_llm_config() -> dict:
    config_path = BASE_DIR / "llm_config.toml"
    if not config_path.exists():
        return {}
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        return data.get("llm", {})
    except Exception:
        return {}

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATABASE_DIR = DATA_DIR / "database"
SQLITE_DB_PATH = DATABASE_DIR / "bike_hotspot.db"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"
MODEL_DIR = OUTPUT_DIR / "models"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH}")

SD_API_URL = os.getenv(
    "SD_API_URL",
    "http://data.sd.gov.cn/gateway/api/1/publicBikeOrder/ls",
)
SD_CLIENT_ID = os.getenv("SD_CLIENT_ID", "")
SD_CLIENT_SECRET = os.getenv("SD_CLIENT_SECRET", "")

_LLM_CONFIG = _load_llm_config()
LLM_API_KEY = os.getenv("LLM_API_KEY", _LLM_CONFIG.get("api_key", ""))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", _LLM_CONFIG.get("base_url", ""))
LLM_MODEL = os.getenv("LLM_MODEL", _LLM_CONFIG.get("model", ""))

LIANGSHAN_CENTER = (116.095, 35.805)
LNG_MIN = float(os.getenv("LNG_MIN", "115.8"))
LNG_MAX = float(os.getenv("LNG_MAX", "116.4"))
LAT_MIN = float(os.getenv("LAT_MIN", "35.5"))
LAT_MAX = float(os.getenv("LAT_MAX", "36.1"))
GRID_SIZE = float(os.getenv("GRID_SIZE", "0.005"))

DEFAULT_INTERVAL_MINUTES = int(os.getenv("DEFAULT_INTERVAL_MINUTES", "30"))
DEFAULT_LOOKBACK_DAYS = int(os.getenv("DEFAULT_LOOKBACK_DAYS", "3"))


def ensure_directories() -> None:
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        DATABASE_DIR,
        FIGURE_DIR,
        TABLE_DIR,
        REPORT_DIR,
        MODEL_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
