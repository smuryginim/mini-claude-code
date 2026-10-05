import json
import re
from pathlib import Path
from typing import List, Dict, Any, Callable

HISTORY_FILE = "history.json"
LOG_COLOR = "\u001b[90m"
RESET_COLOR = "\u001b[0m"


def resolve_abs_path(path_str: str) -> Path:
    """Resolves relative or tilde path strings to absolute Path objects."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def extract_retry_delay(err: Exception, max_allowed: float = 60.0) -> float:
    """Extracts recommended retryDelay seconds from API error responses.
    Caps max delay to max_allowed (default 60s) to prevent absurd waits.
    """
    err_str = str(err)

    # 1. Match 'retry in 49.410388817s'
    match = re.search(r"retry in ([\d\.]+)s", err_str, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1)) + 1.0
            return min(val, max_allowed)
        except ValueError:
            pass

    # 2. Match 'retryDelay': '49s' or "retryDelay": "49s"
    match = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"]([\d\.]+)s?['\"]", err_str, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1)) + 1.0
            return min(val, max_allowed)
        except ValueError:
            pass

    # 3. Fallback generic match
    match = re.search(r"retryDelay[\"':\s]+([\d\.]+)s?", err_str, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1)) + 1.0
            return min(val, max_allowed)
        except ValueError:
            pass

    return 15.0


def load_conversation(prompt_generator_fn: Callable[[], str]) -> List[Dict[str, str]]:
    """Loads conversation history from history.json if available."""
    # Note: Using absolute path from root for history file to maintain consistent state location
    history_path = Path.cwd() / HISTORY_FILE
    if history_path.exists():
        try:
            data = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0:
                print(f"{LOG_COLOR}[Session] Loaded {len(data)} history messages from {HISTORY_FILE}{RESET_COLOR}")
                # Ensure latest system prompt is injected at index 0
                data[0] = {"role": "developer", "content": prompt_generator_fn()}
                return data
        except Exception as e:
            print(f"{LOG_COLOR}[Session] Error reading {HISTORY_FILE}: {e}{RESET_COLOR}")

    return [{
        "role": "developer",
        "content": prompt_generator_fn()
    }]


def save_conversation(conversation: List[Dict[str, str]]):
    """Saves current conversation state to history.json."""
    try:
        (Path.cwd() / HISTORY_FILE).write_text(json.dumps(conversation, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"{LOG_COLOR}[Session] Error saving history: {e}{RESET_COLOR}")
