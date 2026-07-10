from functools import lru_cache
from src.config import CONFIG


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, model_name: str | None = None) -> int:
    if model_name is None:
        model_name = CONFIG.embed_model

    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=True))