"""Storage module for managing candidate solutions."""
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data.json"

def load_candidates() -> list[dict]:
    """Load candidate solutions from JSON file."""
    if not DATA_PATH.exists():
        return []
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))

def save_candidates(candidates: list[dict]) -> None:
    """Save candidate solutions to JSON file."""
    DATA_PATH.write_text(json.dumps(candidates, indent=2), encoding="utf-8") 