# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RetailOptima AI** — a Streamlit web app that scrapes Amazon product pages and generates optimized Polish-language marketing descriptions using local Ollama LLMs, plus AI-enhanced product images via a local FLUX.2-klein-4B model on GPU.

## Running the App

```bash
# Install dependencies (uses uv, CUDA 12.4 build of PyTorch on Linux/Windows)
uv sync

# Start the Streamlit app
uv run streamlit run app.py
```

Ollama must be running locally at `http://localhost:11434` with at least one of these models installed:
- `gemma3:4b` (fast)
- `gpt-oss:20b` (balanced)
- `gemma4:31b` (precise)

Image generation requires CUDA-capable GPU with sufficient VRAM. The app degrades gracefully if `src/image_gen.py` cannot be imported (no GPU or missing dependencies).

## Architecture

```
app.py              ← Streamlit UI + orchestration logic
src/
  scraper.py        ← Amazon HTML scraping (httpx + BeautifulSoup4, JSON-LD preferred)
  llm.py            ← Multi-provider streaming LLM client (Ollama, Gemini, OpenAI, Anthropic)
  image_gen.py      ← FLUX.2-klein-4B img2img pipeline (diffusers, cached in _pipeline_cache)
```

### Data flow

1. User pastes Amazon URL → `scraper.fetch_page_html()` fetches with rotating user-agents
2. `scraper.extract_product_data()` parses title, description, reviews (up to 15), and image URLs
3. `app.detect_context_and_tone()` calls Ollama to pick the best copywriting tone + SEO keywords
4. A structured Polish-language prompt (with `[OPIS]`, `[OPIS_OBRAZU]`, `[PROMPT_1..3]` tags) is streamed via `llm.stream_chat()` to Ollama
5. `app.parse_ollama_output()` extracts the three tagged sections from the raw stream
6. Optionally: Ollama is unloaded from VRAM via `llm.unload_model()`, then FLUX generates an img2img result

### VRAM management

The app shares a single consumer GPU between Ollama and FLUX. Before loading FLUX, `image_gen.clear_gpu_cache()` moves the pipeline to CPU. Before loading Ollama, `llm.unload_model()` sends `keep_alive: 0` to free GPU memory.

### LLM provider abstraction (`src/llm.py`)

`stream_chat()` and `chat()` accept a `provider` argument (`"ollama"`, `"gemini"`, `"openai"`, `"anthropic"`). Only Ollama is wired into the UI today; the other providers are implemented but unused by `app.py`. Each provider has a dedicated `format_messages_for_*()` helper that handles vision (base64 images).

### Scraper resilience

`extract_original_description()` tries five sources in order: Amazon feature bullets → `#productDescription` → JSON-LD schema → meta tags → fallback paragraph scan. The same layered approach applies to titles, reviews, and image URLs. Amazon bot-protection (403/503) raises `ScraperError` immediately.

## Key constraints

- **No tests** — no test suite exists yet.
- **Polish UI** — all user-facing strings in `app.py` are in Polish.
- **Image generation is optional** — wrapped in `try/except` at import time; `HAS_IMAGE_GEN` flag gates all FLUX UI elements.
- **FLUX input constraints** — image dimensions must be divisible by 16; model always uses FP16.
- **Streamlit rendering throttle** — streaming text updates are capped at 10 fps (`now - last_update > 0.1`) to avoid high CPU/network load.
