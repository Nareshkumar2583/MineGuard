import json
import os
from datetime import datetime

LOG_PATH = os.path.join("logs", "forecast_log.json")

def save_event(event_type, details):
    """Save structured events to JSONL (append mode)."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "details": details
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
