
import datetime
import json
import os

LOG_FILE = "audit.log"


def log(action: str, details: dict):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "action": action,
        "details": details,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
