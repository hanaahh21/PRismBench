import json
import requests
from typing import Dict
from .config import OLLAMA_URL, OLLAMA_TIMEOUT
from .prompt_builder import SYSTEM_INSTRUCTIONS
from .response_handler import parse_llm_json

def query_ollama_model(model_name: str, user_prompt: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": json.dumps(SYSTEM_INSTRUCTIONS, indent=2)
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )
    except Exception as e:
        raise RuntimeError(f"Ollama connection failed: {e}")

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama HTTP {response.status_code}: {response.text[:500]}"
        )

    data = response.json()

    if "message" not in data or "content" not in data["message"]:
        raise RuntimeError(
            f"Unexpected Ollama response format: {json.dumps(data)[:500]}"
        )

    raw_text = data["message"]["content"]
    return parse_llm_json(raw_text)


def query_gemma(prompt: str, pr: Dict) -> Dict:
    return query_ollama_model("gemma2:9b-8k", prompt)


def query_llama(prompt: str, pr: Dict) -> Dict:
    return query_ollama_model("llama3.1:8b-8k", prompt)


def query_mistral(prompt: str, pr: Dict) -> Dict:
    return query_ollama_model("mistral:7b-8k", prompt)
