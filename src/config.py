from pathlib import Path
import os
import yaml
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


def load_params() -> dict:
    with open(BASE_DIR / "params.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


PARAMS = load_params()


def get_param(*keys, default=None):
    value = PARAMS
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
        if value is None:
            return default
    return value


def get_env(key: str, default=None, required: bool = False):
    value = os.getenv(key, default)

    if required and (value is None or str(value).strip() == ""):
        raise ValueError(f"Variável de ambiente obrigatória não encontrada ou vazia: {key}")

    return value