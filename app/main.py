import os
import sys
import time
import tomllib
import streamlit as st
import logging

from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from controllers.code_controller import CodeController, CodeControllerConfig  # noqa: E402
from controllers.chat_controller import ChatController, ChatControllerConfig  # noqa: E402
from controllers.embedding_controller import (  # noqa: E402
    EmbeddingController,
    EmbeddingControllerConfig,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s::%(name)s] %(message)s",
)


@dataclass
class AppConfig:
    max_new_tokens: int
    delay: float
    chat_system_prompt: str
    code_system_prompt: str

    k: int
    threshold: float

    refine_method: str = "llm"

    config_path: None | str = None

    def __post_init__(self):
        self.embedding_controller_config = EmbeddingControllerConfig(
            self.k, self.threshold
        )
        self.chat_controller_config = ChatControllerConfig(self.chat_system_prompt)
        self.code_controller_config = CodeControllerConfig(self.code_system_prompt)


class ChatPage:
    MAX_CHAT_NAME_LEN = 50

    def __init__(self, config: AppConfig):
        self.config = config

        if "chat_controller" not in st.session_state:
            st.session_state.chat_controller = ChatController(
                config.chat_controller_config
            )
        if "code_controller" not in st.session_state:
            st.session_state.code_controller = CodeController(
                config.code_controller_config
            )
        if "embedding_controller" not in st.session_state:
            st.session_state.embedding_controller = EmbeddingController(
                config.embedding_controller_config
            )
        if "active_chat_id" not in st.session_state:
            st.session_state.active_chat_id = None

    def _stream_response(self, text: str, delay: float = 0.03):
        placeholder = st.empty()
        full_text = ""
        for word in text.split():
            full_text += word + " "
            time.sleep(delay)
            placeholder.text(full_text + "▌")
        placeholder.markdown(text)
        return text

    @staticmethod
    def _derive_chat_name(prompt: str) -> str:
        """Derive a short chat name from the user's first message."""
        name = prompt.strip().replace("\n", " ")
        if len(name) > ChatPage.MAX_CHAT_NAME_LEN:
            name = name[: ChatPage.MAX_CHAT_NAME_LEN] + "..."
        return name

    def _render_sidebar(self):
        """Render the sidebar with chat history, new chat button, and settings."""
        chat_controller: ChatController = st.session_state.chat_controller

        with st.sidebar:
            st.header("Chats")

            if st.button("New Chat", use_container_width=True):
                chat_controller.new_chat()
                st.session_state.active_chat_id = None
                st.rerun()

            st.divider()

            saved_chats = chat_controller.list_chats()
            if not saved_chats:
                st.caption("No saved chats yet.")
            else:
                for chat in reversed(saved_chats):
                    chat_id = chat["chat_id"]
                    label = chat.get("chat_name", f"Chat {chat_id}")
                    is_active = st.session_state.active_chat_id == chat_id

                    if st.button(
                        label,
                        key=f"chat_{chat_id}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        if not is_active:
                            chat_controller.load_chat(chat_id)
                            st.session_state.active_chat_id = chat_id
                            st.rerun()

            # settings gear at the bottom
            st.divider()
            if st.button("Settings", icon=":material/settings:", use_container_width=True):
                self._open_settings_dialog()

    @st.dialog("Settings", width="large")
    def _open_settings_dialog(self):
        """Password-protected settings dialog for tunable config options."""
        if "settings_authenticated" not in st.session_state:
            st.session_state.settings_authenticated = False

        if not st.session_state.settings_authenticated:
            password = st.text_input(
                "Enter password to access settings",
                type="password",
                key="settings_password",
            )
            if st.button("Unlock"):
                if password == SETTINGS_PASSWORD:
                    st.session_state.settings_authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            return

        st.caption("Changes are saved to `app_config.toml` and take effect on next interaction.")

        max_new_tokens = st.number_input(
            "Max New Tokens",
            min_value=512,
            value=self.config.max_new_tokens,
            step=128,
            key="settings_max_new_tokens",
        )

        delay = st.slider(
            "Response Delay (seconds per word)",
            min_value=0.02,
            max_value=0.07,
            value=self.config.delay,
            step=0.01,
            key="settings_delay",
        )

        k = st.number_input(
            "Top-K Embedding Results",
            min_value=3,
            value=self.config.k,
            step=1,
            key="settings_k",
        )

        threshold = st.slider(
            "Cosine Similarity Threshold",
            min_value=0.3,
            max_value=1.0,
            value=self.config.threshold,
            step=0.05,
            key="settings_threshold",
        )

        refine_options = ["llm", "keyword"]
        refine_method = st.selectbox(
            "Query Refinement Method",
            options=refine_options,
            index=refine_options.index(self.config.refine_method)
            if self.config.refine_method in refine_options
            else 0,
            key="settings_refine_method",
        )

        if st.button("Save", type="primary", use_container_width=True):
            updates = {
                "max_new_tokens": int(max_new_tokens),
                "delay": round(float(delay), 2),
                "k": int(k),
                "threshold": round(float(threshold), 2),
                "refine_method": refine_method,
            }
            save_config_toml(self.config.config_path, updates)

            # update runtime config
            self.config.max_new_tokens = updates["max_new_tokens"]
            self.config.delay = updates["delay"]
            self.config.k = updates["k"]
            self.config.threshold = updates["threshold"]
            self.config.refine_method = updates["refine_method"]

            # update controller configs that depend on these values
            self.config.embedding_controller_config.k = updates["k"]
            self.config.embedding_controller_config.threshold = updates["threshold"]

            st.toast("Settings saved successfully!", icon=":material/check:")
            st.rerun()

    def render(self):
        st.set_page_config(page_title="moviemate", layout="wide")
        st.title("moviemate")

        chat_controller = st.session_state.chat_controller
        code_controller = st.session_state.code_controller
        embedding_controller = st.session_state.embedding_controller

        self._render_sidebar()

        # populate messages
        for message in chat_controller.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # take user prompt/input
        if prompt := st.chat_input("Ask anything movies"):
            st.chat_message("user").markdown(prompt)

            # set chat name from first user message
            is_first_message = all(
                m["role"] == "system" for m in chat_controller.messages()
            )
            if is_first_message:
                chat_controller.chat_name = self._derive_chat_name(prompt)

            # code model returns database data
            db_data = code_controller.sample(
                prompt=prompt,
                max_new_tokens=self.config.max_new_tokens,
                conversation_context=chat_controller.messages(),
            )

            # embedding model returns best matching movie
            best_match = embedding_controller.best_match(
                prompt,
                conversation_context=chat_controller.messages(),
                refine_method=self.config.refine_method,
            )

            # formulate prompt for chat model
            combined_prompt = (
                f"User asks: {prompt} | "
                f"Relevant database information: {db_data} | "
                f"Best matching movie information: {best_match}"
            )

            # generate response using chat model (and display it)
            with st.chat_message("assistant"):
                _ = self._stream_response(
                    chat_controller.generate(
                        query=combined_prompt,
                        max_new_tokens=self.config.max_new_tokens,
                        user_query=prompt,
                    ),
                    delay=self.config.delay,
                )

            # auto-save after each exchange
            saved_path = chat_controller.save_chat()
            st.session_state.active_chat_id = chat_controller.ctx._chat_id
            logging.getLogger(__name__).info(f"Chat saved to {saved_path}")


SETTINGS_PASSWORD = "1234"


def load_config(path: str) -> dict:
    """
    Load TOML config file into a dictionary.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_app_config(path: str) -> AppConfig:
    """
    Create an AppConfig instance directly from a TOML file.
    """
    data = load_config(path)

    return AppConfig(
        max_new_tokens=data.get("max_new_tokens", 0),
        delay=data.get("delay", 0.0),
        chat_system_prompt=data.get("chat_system_prompt", ""),
        code_system_prompt=data.get("code_system_prompt", ""),
        k=data.get("k", 10),
        threshold=data.get("threshold", 0.75),
        refine_method=data.get("refine_method", "keyword"),
        config_path=path,
    )


def save_config_toml(path: str, updates: dict):
    """
    Update tunable keys in the TOML config file while preserving
    the large system prompt strings verbatim.

    Reads the existing file, parses it, applies updates to the
    simple keys, and writes everything back.
    """
    data = load_config(path)
    data.update(updates)

    def _escape_toml_string(s: str) -> str:
        """Wrap a string for TOML output."""
        if "\n" in s:
            return '"""\n' + s + '"""'
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = []
    for key, value in data.items():
        if isinstance(value, str):
            lines.append(f"{key} = {_escape_toml_string(value)}")
        elif isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, int):
            lines.append(f"{key} = {value}")
        elif isinstance(value, float):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f"{key} = {value}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


config = load_app_config("./app/app_config.toml")
page = ChatPage(config)
page.render()
