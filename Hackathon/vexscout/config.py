import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ROBOTEVENTS_TOKEN = os.getenv("ROBOTEVENTS_TOKEN", "").strip()
DB_PATH = Path(os.getenv("VEXSCOUT_DB", ROOT / "data" / "vexscout.db"))
if not DB_PATH.is_absolute():
    DB_PATH = ROOT / DB_PATH
MODEL = os.getenv("CLAUDE_MODEL") or os.getenv("VEXSCOUT_MODEL") or "claude-opus-5-5"
ROBOTEVENTS_BASE_URL = os.getenv("ROBOTEVENTS_BASE_URL", "https://events.vex.com/api/v2").rstrip("/")
MY_TEAM_NUMBER = os.getenv("MY_TEAM_NUMBER", "").strip().upper()
MY_TEAM_ID = int(os.getenv("MY_TEAM_ID") or 0) or None
SEASON_ID = int(os.getenv("SEASON_ID") or 0) or None
RULES_PATH = ROOT / "data" / "game_rules.md"
MODEL_VERSION = "elo+opr-v1"

# Second opinion for AI reports (Claude drafts, ChatGPT reviews, Claude finalises). Optional.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5").strip()

# Third panel agent (Google Gemini). Optional.
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro").strip()
