from pathlib import Path
from dataclasses import dataclass

from context_manager import ContextManager
from models import main_model


@dataclass
class ChatControllerConfig:
    system_prompt: str
    storage_dir: str = "./chat_history"


class ChatController:
    """
    Simple chat controller. Manages context and generation for the chat model.
    Since the underlying model is the same for chat and code, this class will
    handle context for one purpose, chat.

    Once initialized, following functions are useable:
        generate(query, max_new_tokens)
        save_chat()
        load_chat(chat_id)
        list_chats()
        new_chat()
    """

    def __init__(self, config: ChatControllerConfig):
        self.config = config
        self.ctx = ContextManager(
            config.system_prompt,
            storage_dir=config.storage_dir,
        )

    @property
    def messages(self):
        return self.ctx

    @property
    def chat_name(self) -> str:
        return self.ctx.chat_name

    @chat_name.setter
    def chat_name(self, name: str):
        self.ctx.chat_name = name

    def save_chat(self) -> Path:
        return self.ctx.save_context()

    def load_chat(self, chat_id: int):
        self.ctx.load_context(chat_id)

    def list_chats(self) -> list[dict]:
        return self.ctx.list_saved_chats()

    def new_chat(self):
        self.ctx = ContextManager(
            self.config.system_prompt,
            storage_dir=self.config.storage_dir,
        )

    def generate(self, query: str, max_new_tokens: int = 256, user_query: str | None = None) -> str:
        """
        User's query is automatically added to the context. Main model will
        generate `max_new_tokens` number of tokens.

        When user_query is provided, the clean user text is stored in
        persistent context (for display/history), but the augmented query
        (with DB + embedding results) is sent to the model for generation.

        Args:
            query (str): augmented prompt (with retrieved context)
            max_new_tokens (int, default=256): max generated tokens
            user_query (str|None): clean user input for context history
        Returns:
            response (str): model's response given history as string
        """
        print(
            "[INFO::chat_controller::generate] "
            f"received query: {query}. Generating response"
        )

        # store clean user message in persistent context for display
        display_msg = user_query if user_query is not None else query
        self.ctx.add_user(display_msg)

        # build generation context: swap last user message with the
        # augmented query so the model sees DB + embedding results
        if user_query is not None:
            gen_context = self.ctx()[:-1] + [{"role": "user", "content": query}]
        else:
            gen_context = self.ctx()

        response = main_model.generate(
            message_history=gen_context,
            max_new_tokens=max_new_tokens,
        )

        self.ctx.add_model(response)

        print(
            "[INFO::chat_controller::generate] "
            f"Generated response: '{response}'"
        )

        return response
