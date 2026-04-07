import re
import json

from psycopg.abc import QueryNoTemplate
from typing import Any
from dataclasses import dataclass

from data.data_engine import query_db
from context_manager import ContextManager
from models import main_model


@dataclass
class CodeControllerConfig:
    system_prompt: str


class CodeController:
    """
    Simple code controller. Manages context and generation for the code model.
    Since the underlying model is the same for chat and code, this class will
    handle context for one purpose, code.

    Once initialized, following functions are useable:
        generate(query, max_new_tokens)
    """

    def __init__(self, config: CodeControllerConfig):
        self.config = config
        self.ctx = ContextManager(config.system_prompt)

    @property
    def messages(self):
        return self.ctx

    def generate(
        self,
        query: str,
        max_new_tokens: int = 256,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> str:
        """
        User's query is automatically added to the context. Main model will
        generate `max_new_tokens` number of tokens.

        When conversation_context is provided, recent chat history is
        included so the code model can resolve multi-turn references
        (e.g. "same director", "something more recent", "that one").

        Args:
            query (str): user prompt/query
            max_new_tokens (int, default=256): max generated tokens
            conversation_context (list[dict]): chat history for multi-turn
        Returns:
            response (str): model's response given history as string
        """
        print(f"[INFO::code_controller::generate] received query: {query}")

        # build generation context: code system prompt + recent chat
        # history (for multi-turn reference resolution) + current query
        gen_context = [self.ctx()[0]]  # code system prompt

        if conversation_context:
            recent = [
                m for m in conversation_context
                if m["role"] in ("user", "assistant")
            ][-6:]
            if recent:
                history_text = "\n".join(
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in recent
                )
                gen_context.append({
                    "role": "user",
                    "content": (
                        f"[CONVERSATION HISTORY]\n{history_text}\n"
                        f"[END CONVERSATION HISTORY]\n\n"
                        f"Based on the conversation above, generate SQL for: {query}"
                    ),
                })
            else:
                gen_context.append({"role": "user", "content": query})
        else:
            gen_context.append({"role": "user", "content": query})

        response = main_model.generate(
            message_history=gen_context,
            max_new_tokens=max_new_tokens,
        )

        print(f"[INFO::code_controller::generate] Generated response: '{response}'")

        return response

    def sample(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        conversation_context: list[dict[str, str]] | None = None,
    ):
        response = self.generate(prompt, max_new_tokens, conversation_context)

        if response.strip() == "Sorry I can not help you with that.":
            return response

        query, params = _safe_extract(response)

        if query is None:
            return "I could not generate a valid database query for that request."

        if not self._is_safe_query(query):  # pyright: ignore
            return "Unsafe query blocked."

        results = query_db(
            query=query,
            params=params,
        )

        return self._parse(results)

    # def _safe_extract(
    #     self,
    #     model_output: str,
    # ) -> tuple[QueryNoTemplate | None, tuple[Any, ...]]:
    #     """
    #     Extract query and params from code-model's output.
    #
    #     Expects model to return JSON:
    #     {
    #         "query": "... %s ...",
    #         "params": [...]
    #     }
    #
    #     Args:
    #         model_output (str): model's output
    #     Returns:
    #         tuple[QueryNoTemplate, tuple[Any, ...]]: tuple containing
    #         query and params
    #     """
    #     try:
    #         data = json.loads(model_output)
    #         query = data.get("query")
    #         params = tuple(data.get("params", []))
    #
    #         if not isinstance(query, str):
    #             return None, ()
    #         if query.strip() == "":
    #             return None, ()
    #         return query, params  # pyright: ignore
    #
    #     except Exception:
    #         return None, ()
    def _safe_extract(
        self,
        model_output: str,
    ) -> tuple[QueryNoTemplate | None, tuple[Any, ...]]:
        """
        Robustly extract query and params from Qwen3's output.

        Handles:
          - <think>...</think> blocks (Qwen3 reasoning traces)
          - ```json ... ``` or ``` ... ``` markdown fences
          - Leading/trailing whitespace and stray text before/after JSON
          - Partial JSON at end of output (truncated generation)
        """
        text = model_output

        # 1. Strip Qwen3 <think> blocks (may be multiline)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if fence_match:
            text = fence_match.group(1)

        # 3. Extract the first {...} JSON object from whatever remains
        #    (guards against stray prose before or after the JSON)
        json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not json_match:
            return None, ()

        raw_json = json_match.group(0)

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            return None, ()

        query = data.get("query")
        params = data.get("params", [])

        if not isinstance(query, str) or query.strip() == "":
            return None, ()

        if not isinstance(params, list):
            params = []

        return query, tuple(params)  # pyright: ignore

    def _is_safe_query(self, query: str) -> bool:
        """Allow read-only SELECT queries."""
        query = query.strip().lower()

        if not query.startswith("select"):
            return False

        forbidden = ["insert", "update", "delete", "drop", "alter", "truncate"]

        return not any(word in query for word in forbidden)

    def _parse(self, results: list[tuple]) -> str:
        """
        Convert DB output into a simple string for the chat model.
        """
        if not results:
            return "No results found."

        lines = []
        for row in results:
            row_str = ", ".join(str(col) for col in row)
            lines.append(row_str)

        MAX_ROWS = 20
        lines = lines[:MAX_ROWS]

        output = "\n".join(lines)

        if len(lines) > MAX_ROWS:
            output += f"\n ... ({len(lines) - MAX_ROWS} more rows)."

        return output


def _sanitize_interpolated_wildcards(query: str, params: list) -> tuple[str, list]:
    """
    Catch the model's most common mistake: putting wildcard values directly
    into the query string instead of using %s placeholders.

    Example of bad model output:
        query:  "WHERE director ILIKE '%Christopher Nolan%' AND ..."
        params: ["%Christopher Nolan%", ...]

    This function detects quoted wildcard literals in the query string
    (e.g. '%somevalue%', '%somevalue', 'somevalue%') and replaces each
    with a %s placeholder, appending the extracted value to params in the
    correct positional order.

    Only fixes ILIKE/LIKE patterns — other quoted strings are left alone.
    """
    # Match: ILIKE/LIKE followed by a single-quoted string containing %
    pattern = re.compile(
        r"(I?LIKE\s*)'(%?[^']+%?)'",
        flags=re.IGNORECASE,
    )

    extracted = []

    def replacer(m: re.Match) -> str:
        extracted.append(m.group(2))  # the '%value%' string
        return f"{m.group(1)}%s"  # replace with placeholder

    sanitized_query = pattern.sub(replacer, query)

    if not extracted:
        return query, params  # nothing to fix

    # Rebuild params: the sanitizer extracted values that the model had
    # already put in params too (duplicate) — or it hadn't included them.
    # Strategy: recount %s in the sanitized query and reconcile.
    expected_count = sanitized_query.count("%s")
    combined = extracted + [p for p in params if p not in extracted]
    reconciled = combined[:expected_count]

    return sanitized_query, reconciled


def _safe_extract(
    model_output: str,
) -> tuple[QueryNoTemplate | None, tuple[Any, ...]]:
    """
    Robustly extract query and params from Qwen3's output.

    Handles:
      - <think>...</think> blocks (Qwen3 reasoning traces)
      - ```json ... ``` or ``` ... ``` markdown fences
      - Leading/trailing whitespace and stray text before/after JSON
      - Partial JSON at end of output (truncated generation)
      - Interpolated wildcard literals in the query string (e.g. ILIKE '%value%')
    """
    text = model_output

    # 1. Strip Qwen3 <think> blocks (may be multiline)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # 3. Extract the first {...} JSON object from whatever remains
    #    (guards against stray prose before or after the JSON)
    json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not json_match:
        return None, ()

    raw_json = json_match.group(0)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None, ()

    query = data.get("query")
    params = data.get("params", [])

    if not isinstance(query, str) or query.strip() == "":
        return None, ()

    if not isinstance(params, list):
        params = []

    # 4. Fix any interpolated wildcard literals the model snuck into the query
    query, params = _sanitize_interpolated_wildcards(query, params)

    # 5. Validate placeholder count matches param count
    placeholder_count = query.count("%s")
    if placeholder_count != len(params):
        return None, ()

    return query, tuple(params)  # pyright: ignore
