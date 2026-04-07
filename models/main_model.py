"""
Chat and code model implementation.
Model contexts will be managed externally.
This is purely for model interations, inputs, outputs.
"""

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from torchinfo import summary

from utils import auto_device_map

# load model
device = auto_device_map()
model_id = "Qwen/Qwen2.5-7B-Instruct"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
# )

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config,
)
model = torch.compile(model, backend="inductor")
for p in model.parameters():  # pyright: ignore
    p.requires_grad_ = False

print(f"[INFO::main_model] loaded model. model.device: {model.device}")  # pyright: ignore
print("[INFO::main_model]\n")
summary(model)  # pyright: ignore


@torch.inference_mode()
def generate(
    message_history: list[dict[str, str]],
    max_new_tokens: int = 256,
) -> str:
    """
    Given message history, generate max_new_tokens number of tokens
    from model and return them as a string.

    Args:
        message_history (list[dict[str, str]]): message history
        max_new_tokens (int, defauly=256): max new tokens to generate
    Returns:
        str, decoded result/output from model
    """
    inputs = tokenizer.apply_chat_template(
        message_history,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)  # pyright: ignore
    out = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return out
