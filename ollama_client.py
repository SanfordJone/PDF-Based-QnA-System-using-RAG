import os

import requests


class OllamaError(RuntimeError):
    """Raised when the Ollama server cannot be reached or returns an error."""


class OllamaClient:
    """
    Client for interacting with a local Ollama server (default model: Llama 3.1)
    """

    def __init__(self, base_url=None, model=None, timeout=120):
        """
        Args:
            base_url (str): Ollama API base URL. Defaults to the OLLAMA_HOST
                environment variable, then http://localhost:11434
            model (str): Model name. Defaults to OLLAMA_MODEL, then llama3.1
            timeout (int): Seconds to wait for a response before giving up
        """
        self.base_url = (base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self.timeout = timeout

    def is_available(self):
        """Return True if the Ollama server answers."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return response.ok
        except requests.exceptions.RequestException:
            return False

    def chat(self, messages, temperature=0.2, max_tokens=512):
        """
        Get a single, non-streamed reply from the Ollama chat endpoint

        Args:
            messages (list): Message dicts with 'role' and 'content'
            temperature (float): Sampling temperature (low keeps answers close to the context)
            max_tokens (int): Maximum tokens to generate

        Returns:
            str: The model's reply text
        """
        payload = {
            "model": self.model,
            "messages": messages,
            # Without this Ollama streams many JSON lines and response.json() fails
            "stream": False,
            # Ollama reads generation settings from "options", not the top level
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Could not reach Ollama at {self.base_url}: {e}") from e
        except ValueError as e:
            raise OllamaError("Ollama returned a response that was not valid JSON") from e

        content = (data.get("message") or {}).get("content", "").strip()
        if not content:
            raise OllamaError("Ollama returned an empty answer")
        return content

    def list_models(self):
        """Return the names of models installed in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return [m.get("name") for m in response.json().get("models", [])]
        except (requests.exceptions.RequestException, ValueError):
            return []
