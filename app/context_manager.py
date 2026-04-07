import json

from pathlib import Path
from datetime import datetime


# Index helpers
def _load_index(storage_dir: Path) -> dict:
    index_path = storage_dir / "index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"chats": {}}


def _save_index(storage_dir: Path, index: dict) -> None:
    index_path = storage_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def _next_chat_id(index: dict) -> int:
    """Return the next available integer chat ID."""
    existing = [int(k) for k in index["chats"].keys()] if index["chats"] else [0]
    return max(existing) + 1


class ContextManager:
    """
    Re-written by Claude for context persistence.

    Manages conversation history for a single chat session.

    Persistence
    ~~~~~~~~~~~
    Call save_context() to write the session to disk.
    Call load_context(chat_id) to restore a previous session from disk.

    Both methods require storage_dir to be set at construction time.

    Args:
        system_prompt (str)  : The system prompt for this session.
        storage_dir   (str)  : Directory for persisted chat files.
                               Defaults to "./chat_history".
        chat_name     (str)  : Human-readable label for this session.
                               Defaults to a timestamp-based name.
        max_chats     (int)  : Maximum number of saved chats to keep in the
                               index before oldest entries are pruned.
                               Defaults to 100.
    """

    def __init__(
        self,
        system_prompt: str,
        storage_dir: str = "./chat_history",
        chat_name: str | None = None,
        max_chats: int = 100,
    ):
        self._system_prompt = system_prompt
        self._context: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        self._storage_dir = Path(storage_dir)
        self._max_chats = max_chats
        self._created_at = datetime.now().isoformat(timespec="seconds")
        self._chat_name = chat_name or f"Chat {self._created_at}"

        # Assigned when saved for the first time
        self._chat_id: int | None = None
        self._filename: str | None = None

    # ── Core context methods ───────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        self._context.append({"role": "user", "content": content})

    def add_model(self, content: str) -> None:
        self._context.append({"role": "assistant", "content": content})

    @property
    def context(self) -> list[dict[str, str]]:
        return self._context

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def chat_name(self) -> str:
        return self._chat_name

    @chat_name.setter
    def chat_name(self, name: str) -> None:
        self._chat_name = name

    def __iter__(self):
        return iter(self._context)

    def __call__(self) -> list[dict[str, str]]:
        return self._context

    # ── Turn count ─────────────────────────────────────────────────────────────

    def _turn_count(self) -> int:
        """Number of completed user+assistant turns (excludes system message)."""
        return sum(1 for m in self._context if m["role"] in ("user", "assistant")) // 2

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_context(self) -> Path:
        """
        Persist the current session to disk.

        On the first call a new chat ID is allocated and registered in
        index.json. Subsequent calls update the existing file in place.
        If the index has reached max_chats, the oldest entry is removed
        from the index (the file itself is not deleted).

        Returns:
            Path: absolute path to the saved chat file.
        """
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        index = _load_index(self._storage_dir)

        # Allocate an ID if this is the first save
        if self._chat_id is None:
            self._chat_id = _next_chat_id(index)
            self._filename = f"chat_{self._chat_id}.json"

            # Prune oldest entry if we have hit the cap
            if len(index["chats"]) >= self._max_chats:
                oldest_key = min(index["chats"].keys(), key=lambda k: int(k))
                del index["chats"][oldest_key]

            index["chats"][str(self._chat_id)] = self._filename
            _save_index(self._storage_dir, index)

        chat_path = self._storage_dir / self._filename

        payload = {
            "metadata": {
                "chat_id": self._chat_id,
                "chat_name": self._chat_name,
                "filename": self._filename,
                "created_at": self._created_at,
                "last_saved_at": datetime.now().isoformat(timespec="seconds"),
                "system_prompt": self._system_prompt,
                "turn_count": self._turn_count(),
            },
            "dialogue": self._context,
        }

        with chat_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[ContextManager] Session saved -> {chat_path}  (id={self._chat_id})")
        return chat_path.resolve()

    def load_context(self, chat_id: int) -> None:
        """
        Restore a previously saved session into this instance.

        Replaces the current context, system prompt, chat name, and
        metadata with those from the saved file. The session's original
        created_at timestamp is preserved; last_saved_at is not touched
        until the next save_context() call.

        Args:
            chat_id (int): The integer ID shown in index.json.

        Raises:
            FileNotFoundError : If index.json or the chat file is missing.
            KeyError          : If chat_id is not in the index.
            ValueError        : If the chat file is malformed.
        """
        index = _load_index(self._storage_dir)

        key = str(chat_id)
        if key not in index["chats"]:
            raise KeyError(
                f"[ContextManager] Chat ID {chat_id} not found in index. "
                f"Available IDs: {sorted(int(k) for k in index['chats'])}"
            )

        filename = index["chats"][key]
        chat_path = self._storage_dir / filename

        if not chat_path.exists():
            raise FileNotFoundError(
                f"[ContextManager] Chat file not found: {chat_path}"
            )

        with chat_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if "metadata" not in payload or "dialogue" not in payload:
            raise ValueError(f"[ContextManager] Malformed chat file: {chat_path}")

        meta = payload["metadata"]
        self._chat_id = meta["chat_id"]
        self._filename = meta["filename"]
        self._chat_name = meta["chat_name"]
        self._created_at = meta["created_at"]
        self._system_prompt = meta["system_prompt"]
        self._context = payload["dialogue"]

        print(
            f"[ContextManager] Session loaded <- {chat_path}  "
            f"(id={self._chat_id}, turns={meta['turn_count']}, "
            f"name='{self._chat_name}')"
        )

    # ── Index utilities ────────────────────────────────────────────────────────

    def list_saved_chats(self) -> list[dict]:
        """
        Return a summary of all saved chats from the index.

        Each entry contains: chat_id, filename, chat_name, created_at,
        last_saved_at, turn_count.

        Returns:
            list[dict]: Sorted by chat_id ascending. Empty list if no chats
                        have been saved yet.
        """
        index = _load_index(self._storage_dir)
        summaries = []

        for key, filename in index["chats"].items():
            chat_path = self._storage_dir / filename
            entry = {"chat_id": int(key), "filename": filename}

            if chat_path.exists():
                with chat_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                meta = payload.get("metadata", {})
                entry.update(
                    {
                        "chat_name": meta.get("chat_name", ""),
                        "created_at": meta.get("created_at", ""),
                        "last_saved_at": meta.get("last_saved_at", ""),
                        "turn_count": meta.get("turn_count", 0),
                    }
                )
            else:
                entry.update(
                    {
                        "chat_name": "(file missing)",
                        "created_at": "",
                        "last_saved_at": "",
                        "turn_count": 0,
                    }
                )

            summaries.append(entry)

        return sorted(summaries, key=lambda e: e["chat_id"])
