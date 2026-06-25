import json
import logging
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Minimal Ollama API client.

    Parameters
    ----------
    base_url : str
        Base URL of the Ollama server, e.g. "http://ollama:11434"
    timeout : float
        Request timeout in seconds (default 120 — long chains can be slow)
    """

    def __init__(self, base_url: str = "http://ollama:11434", timeout: float = 120.0):
        # Normalise: strip trailing slash once here
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_chat(
        self,
        model: str,
        messages: list,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        seed: int = -1,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
    ) -> str:
        """
        Send a chat completion request to Ollama.

        Returns the assistant's reply as a plain string.
        On failure returns an "Error: ..." string so callers can detect it.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,          # keep it simple; streaming handled by OWUI itself
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "seed": seed,
                "top_k": top_k,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
            },
        }

        url = f"{self.base_url}/api/chat"
        try:
            r = httpx.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return self._parse_response(r.text)
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama HTTP error %s: %s", exc.response.status_code, exc)
            return f"Error: HTTP {exc.response.status_code} from Ollama"
        except httpx.RequestError as exc:
            logger.error("Ollama request error: %s", exc)
            return f"Error: Could not reach Ollama at {self.base_url} — {exc}"

    def fetch_models(self) -> list:
        """Return a list of model names available on this Ollama server."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except httpx.HTTPError:
            return []

    def unload_model(self, model: str) -> None:
        """Ask Ollama to evict a model from VRAM."""
        payload = {"model": model, "prompt": "", "keep_alive": 0}
        try:
            httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
        except httpx.HTTPError:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_response(self, response_text: str) -> str:
        """
        Parse Ollama's /api/chat response.

        With stream=False the entire response is one JSON object.
        We handle the streaming-line format as a fallback just in case.
        """
        response_text = response_text.strip()

        # Preferred path: single JSON object (stream=False)
        try:
            obj = json.loads(response_text)
            content = obj.get("message", {}).get("content", "")
            if content:
                return content
        except json.JSONDecodeError:
            pass

        # Fallback: newline-delimited stream format
        content = ""
        for line in response_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                chunk = obj.get("message", {}).get("content", "")
                if chunk:
                    content += chunk
            except json.JSONDecodeError:
                continue

        return content or response_text  # last resort: return raw
