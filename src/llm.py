from __future__ import annotations

import json
import re
from typing import Any, Generator

import httpx


class LLMError(Exception):
    pass


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def parse_json_payload(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Regex fallback to find first JSON object
    match = re.search(r"\{.*\}", value, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def list_local_ollama_models(ollama_url: str) -> list[str]:
    """Fetches a list of installed models from local Ollama instance."""
    endpoint = ollama_url.rstrip("/") + "/api/tags"
    try:
        response = httpx.get(endpoint, timeout=3.0)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            return sorted(models)
    except Exception:
        pass
    return []


def format_messages_for_gemini(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Formats standard messages for Gemini API structure, extracting system prompt."""
    gemini_contents = []
    system_instruction = ""

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        images = msg.get("images", [])

        if role == "system":
            system_instruction = content
            continue

        parts = []
        if content:
            parts.append({"text": content})

        for img_base64 in images:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img_base64
                }
            })

        # Gemini uses 'model' instead of 'assistant'
        gemini_role = "model" if role == "assistant" else "user"
        gemini_contents.append({
            "role": gemini_role,
            "parts": parts
        })

    return gemini_contents, system_instruction


def format_messages_for_anthropic(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Formats standard messages for Anthropic API structure, extracting system prompt and handling images."""
    anthropic_messages = []
    system_instruction = ""

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        images = msg.get("images", [])

        if role == "system":
            system_instruction = content
            continue

        parts = []
        for img_base64 in images:
            parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_base64
                }
            })

        if content:
            parts.append({
                "type": "text",
                "text": content
            })

        anthropic_messages.append({
            "role": role,
            "content": parts if parts else content
        })

    return anthropic_messages, system_instruction


def format_messages_for_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Formats standard messages for OpenAI API, embedding base64 images if present."""
    openai_messages = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        images = msg.get("images", [])

        if not images:
            openai_messages.append({"role": role, "content": content})
            continue

        # For vision models in OpenAI, content must be a list of parts
        parts = [{"type": "text", "text": content}]
        for img_base64 in images:
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        openai_messages.append({"role": role, "content": parts})

    return openai_messages


def stream_chat(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str = "",
    ollama_url: str = "http://localhost:11434",
    as_json: bool = False,
    timeout_sec: float = 180.0,
    think: bool | None = None,
) -> Generator[str, None, None]:
    """Streams chat completions from various providers (Ollama, Gemini, OpenAI, Anthropic)."""
    provider = provider.lower()

    # --- OLLAMA ---
    if provider == "ollama":
        endpoint = ollama_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if as_json:
            payload["format"] = "json"
        if think is not None:
            payload["think"] = think

        try:
            with httpx.stream("POST", endpoint, json=payload, timeout=timeout_sec) as response:
                if response.status_code != 200:
                    raise LLMError(f"Ollama zwróciła status {response.status_code}")
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue
        except httpx.HTTPError as exc:
            raise LLMError(f"Błąd połączenia z Ollama: {exc}")

    # --- GEMINI ---
    elif provider == "gemini":
        if not api_key:
            raise LLMError("Brak klucza API dla Gemini. Ustaw go w panelu bocznym lub w zmiennych środowiskowych.")

        # Gemini v1beta endpoint
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={api_key}"
        gemini_contents, system_instruction = format_messages_for_gemini(messages)

        payload: dict[str, Any] = {
            "contents": gemini_contents,
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if as_json:
            payload["generationConfig"] = {
                "responseMimeType": "application/json"
            }

        try:
            with httpx.stream("POST", endpoint, json=payload, timeout=timeout_sec) as response:
                if response.status_code != 200:
                    error_msg = response.read().decode("utf-8")
                    raise LLMError(f"Gemini API zwróciła status {response.status_code}: {error_msg}")
                
                # Gemini returns JSON array chunks containing candidates
                buffer = ""
                for chunk in response.iter_text():
                    buffer += chunk
                    # Simple JSON stream parsing for Gemini chunks
                    # Since Gemini returns text in SSE-like or bracketed format, we look for parts text
                    # Alternatively, we can accumulate and parse, but streaming requires yielding
                    # We can use regex to extract all "text" values in the chunk
                    matches = re.findall(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
                    for match in matches:
                        # Decode escape characters in string
                        try:
                            decoded = json.loads(f'"{match}"')
                            yield decoded
                        except Exception:
                            yield match

        except httpx.HTTPError as exc:
            raise LLMError(f"Błąd połączenia z Gemini API: {exc}")

    # --- OPENAI ---
    elif provider == "openai":
        if not api_key:
            raise LLMError("Brak klucza API dla OpenAI. Ustaw go w panelu bocznym lub w zmiennych środowiskowych.")

        endpoint = "https://api.openai.com/v1/chat/completions"
        openai_messages = format_messages_for_openai(messages)
        
        payload: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "stream": True,
        }
        if as_json:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            with httpx.stream("POST", endpoint, json=payload, headers=headers, timeout=timeout_sec) as response:
                if response.status_code != 200:
                    error_msg = response.read().decode("utf-8")
                    raise LLMError(f"OpenAI API zwróciła status {response.status_code}: {error_msg}")
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPError as exc:
            raise LLMError(f"Błąd połączenia z OpenAI API: {exc}")

    # --- ANTHROPIC ---
    elif provider == "anthropic":
        if not api_key:
            raise LLMError("Brak klucza API dla Anthropic. Ustaw go w panelu bocznym lub w zmiennych środowiskowych.")

        endpoint = "https://api.anthropic.com/v1/messages"
        anthropic_messages, system_instruction = format_messages_for_anthropic(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": 4000,
            "stream": True,
        }
        if system_instruction:
            payload["system"] = system_instruction

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        try:
            with httpx.stream("POST", endpoint, json=payload, headers=headers, timeout=timeout_sec) as response:
                if response.status_code != 200:
                    error_msg = response.read().decode("utf-8")
                    raise LLMError(f"Anthropic API zwróciła status {response.status_code}: {error_msg}")
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    # Anthropic SSE format uses event: and data:
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                            event_type = data.get("type")
                            if event_type == "content_block_delta":
                                chunk = data.get("delta", {}).get("text", "")
                                if chunk:
                                    yield chunk
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPError as exc:
            raise LLMError(f"Błąd połączenia z Anthropic API: {exc}")

    else:
        raise LLMError(f"Nieznany dostawca LLM: {provider}")


def chat(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str = "",
    ollama_url: str = "http://localhost:11434",
    as_json: bool = False,
    timeout_sec: float = 180.0,
    think: bool | None = None,
) -> str:
    """Synchronous chat function that aggregates the stream."""
    chunks = []
    for chunk in stream_chat(
        provider=provider,
        model=model,
        messages=messages,
        api_key=api_key,
        ollama_url=ollama_url,
        as_json=as_json,
        timeout_sec=timeout_sec,
        think=think,
    ):
        chunks.append(chunk)
    return "".join(chunks)


def unload_model(model: str, ollama_url: str = "http://localhost:11434") -> None:
    """Sends a request to Ollama to immediately unload the specified model from GPU memory."""
    endpoint = ollama_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [],
        "keep_alive": 0
    }
    try:
        # Use a short timeout so it doesn't block the UI
        httpx.post(endpoint, json=payload, timeout=5.0)
    except Exception:
        # Fallback to /api/generate if /api/chat with empty messages is not accepted
        fallback_endpoint = ollama_url.rstrip("/") + "/api/generate"
        payload_fallback = {
            "model": model,
            "keep_alive": 0
        }
        try:
            httpx.post(fallback_endpoint, json=payload_fallback, timeout=5.0)
        except Exception:
            pass
