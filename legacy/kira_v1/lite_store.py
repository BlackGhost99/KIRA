import datetime
import json
import os

import config


def _default_store():
    return {
        "preferences": {},
        "todos": [],
        "notes": [],
        "history": [],
    }


def _store_path():
    return getattr(config, "LITE_STORE_FILE", "lite_store.json")


def load_store():
    path = _store_path()
    if not os.path.exists(path):
        return _default_store()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _default_store()
        for key in ["preferences", "todos", "notes", "history"]:
            data.setdefault(key, [] if key != "preferences" else {})
        return data
    except Exception:
        return _default_store()


def save_store(store: dict):
    path = _store_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def add_history(store: dict, question: str, response: str):
    history = store.setdefault("history", [])
    history.append(
        {
            "q": question,
            "r": response,
            "ts": datetime.datetime.now().isoformat(),
        }
    )
    max_items = getattr(config, "LITE_HISTORY_MAX", 100)
    if isinstance(max_items, int) and max_items > 0 and len(history) > max_items:
        store["history"] = history[-max_items:]


def get_recent_history(store: dict, limit: int = 3):
    history = store.get("history", [])
    return history[-limit:] if history else []


def add_todo(store: dict, text: str):
    todos = store.setdefault("todos", [])
    todos.append(
        {
            "text": text.strip(),
            "done": False,
            "created": datetime.datetime.now().isoformat(),
        }
    )


def list_todos(store: dict, include_done: bool = False):
    todos = store.get("todos", [])
    if include_done:
        return todos
    return [t for t in todos if not t.get("done")]


def complete_todo(store: dict, index: int):
    todos = store.get("todos", [])
    if index < 1 or index > len(todos):
        return False
    todos[index - 1]["done"] = True
    todos[index - 1]["done_at"] = datetime.datetime.now().isoformat()
    return True


def add_note(store: dict, text: str):
    notes = store.setdefault("notes", [])
    notes.append(
        {
            "text": text.strip(),
            "created": datetime.datetime.now().isoformat(),
        }
    )


def list_notes(store: dict, limit: int = 5):
    notes = store.get("notes", [])
    return notes[-limit:] if notes else []


def set_preference(store: dict, key: str, value: str):
    prefs = store.setdefault("preferences", {})
    prefs[key.strip()] = value.strip()


def get_preferences(store: dict):
    return store.get("preferences", {})
