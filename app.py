from __future__ import annotations

import base64
import json
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
import streamlit as st
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gpt-oss:20b"
MAX_EXTRACTED_REVIEWS = 200
MAX_REVIEWS_FOR_MODEL = 40


class AppError(Exception):
    pass


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = clean_text(str(value))
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def is_valid_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = clean_text(str(value)) if value is not None else ""
        if text:
            return text
    return ""


def truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def fetch_page_html(url: str, timeout_sec: float = 25.0) -> tuple[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.7"}
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout_sec) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AppError(f"Nie udało się pobrać strony: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        raise AppError("Podany URL nie wskazuje na stronę HTML produktu.")

    return response.text, str(response.url)


def parse_json_ld(soup: BeautifulSoup) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def add_payload(payload: Any) -> None:
        if isinstance(payload, list):
            for item in payload:
                add_payload(item)
            return
        if not isinstance(payload, dict):
            return

        graph = payload.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                add_payload(item)

        entries.append(payload)

    scripts = soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)})
    for script in scripts:
        raw = script.string or script.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        add_payload(payload)

    return entries


def find_product_schema(entries: list[dict[str, Any]]) -> dict[str, Any]:
    for item in entries:
        raw_type = item.get("@type")
        types = raw_type if isinstance(raw_type, list) else [raw_type]
        for value in types:
            if isinstance(value, str) and value.lower() == "product":
                return item
    return {}


def extract_specs(soup: BeautifulSoup) -> dict[str, str]:
    specs: dict[str, str] = {}

    for row in soup.select("table tr")[:150]:
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        key = clean_text(cells[0].get_text(" ", strip=True))
        val = clean_text(cells[1].get_text(" ", strip=True))
        if key and val and key not in specs:
            specs[key] = val
        if len(specs) >= 20:
            return specs

    if specs:
        return specs

    dts = soup.select("dl dt")[:80]
    for dt in dts:
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        key = clean_text(dt.get_text(" ", strip=True))
        val = clean_text(dd.get_text(" ", strip=True))
        if key and val and key not in specs:
            specs[key] = val
        if len(specs) >= 20:
            break

    if specs:
        return specs

    for li in soup.select("li")[:200]:
        text = clean_text(li.get_text(" ", strip=True))
        if ":" not in text:
            continue
        key, val = text.split(":", 1)
        key = clean_text(key)
        val = clean_text(val)
        if key and val and len(key) <= 50 and len(val) <= 200 and key not in specs:
            specs[key] = val
        if len(specs) >= 20:
            break

    return specs


def extract_reviews(soup: BeautifulSoup, product_schema: dict[str, Any]) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []

    raw_reviews = product_schema.get("review")
    if isinstance(raw_reviews, dict):
        raw_reviews = [raw_reviews]
    if isinstance(raw_reviews, list):
        for item in raw_reviews:
            if not isinstance(item, dict):
                continue
            author = item.get("author")
            if isinstance(author, dict):
                author = author.get("name")
            rating = ""
            rating_obj = item.get("reviewRating")
            if isinstance(rating_obj, dict):
                rating = clean_text(str(rating_obj.get("ratingValue", "")))
            text = clean_text(item.get("reviewBody"))
            if text:
                collected.append(
                    {
                        "author": clean_text(str(author)) or "użytkownik",
                        "rating": rating,
                        "text": text,
                    }
                )

    agg = product_schema.get("aggregateRating")
    if isinstance(agg, dict):
        rating_value = clean_text(str(agg.get("ratingValue", "")))
        rating_count = clean_text(str(agg.get("reviewCount", "")))
        if rating_value or rating_count:
            collected.append(
                {
                    "author": "agregat",
                    "rating": rating_value,
                    "text": f"Średnia ocen: {rating_value} ({rating_count} opinii)",
                }
            )

    if len(collected) < 3:
        nodes = soup.select('[class*="review" i], [id*="review" i], [data-review]')
        for node in nodes[:35]:
            text = clean_text(node.get_text(" ", strip=True))
            if 30 <= len(text) <= 700:
                collected.append({"author": "użytkownik", "rating": "", "text": text})

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in collected:
        key = item.get("text", "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= 12:
            break

    return deduped


def extract_images(soup: BeautifulSoup, base_url: str, product_schema: dict[str, Any]) -> list[str]:
    urls: list[str] = []

    def add_url(candidate: str | None) -> None:
        value = clean_text(candidate)
        if not value:
            return
        if value.startswith("data:"):
            return
        full = urljoin(base_url, value)
        if full not in urls:
            urls.append(full)

    meta_og = soup.find("meta", property="og:image")
    if meta_og and meta_og.get("content"):
        add_url(meta_og.get("content"))

    schema_images = product_schema.get("image")
    if isinstance(schema_images, str):
        add_url(schema_images)
    elif isinstance(schema_images, list):
        for item in schema_images:
            if isinstance(item, str):
                add_url(item)
            elif isinstance(item, dict):
                add_url(item.get("url"))

    for img in soup.select("img")[:40]:
        add_url(img.get("src") or img.get("data-src"))

    return urls[:5]


def extract_original_description(soup: BeautifulSoup, product_schema: dict[str, Any]) -> str:
    meta_desc = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    og_desc = soup.find("meta", property="og:description")

    desc = first_non_empty(
        product_schema.get("description") if isinstance(product_schema, dict) else "",
        meta_desc.get("content") if meta_desc else "",
        og_desc.get("content") if og_desc else "",
    )
    if desc:
        return truncate_text(desc, 2200)

    paragraphs: list[str] = []
    for p in soup.select("article p, main p, p")[:40]:
        text = clean_text(p.get_text(" ", strip=True))
        if len(text) >= 60:
            paragraphs.append(text)
        if len(paragraphs) >= 3:
            break

    return truncate_text("\n\n".join(paragraphs), 2200)


def extract_page_excerpt(soup: BeautifulSoup) -> str:
    chunks: list[str] = []
    for node in soup.select("h1, h2, h3, p, li")[:140]:
        text = clean_text(node.get_text(" ", strip=True))
        if len(text) < 20:
            continue
        chunks.append(text)
        if sum(len(x) for x in chunks) >= 4000:
            break
    return truncate_text("\n".join(chunks), 4000)


def extract_product_data(html: str, base_url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    jsonld_entries = parse_json_ld(soup)
    product_schema = find_product_schema(jsonld_entries)

    h1 = soup.find("h1")
    title_tag = soup.find("title")
    og_title = soup.find("meta", property="og:title")

    title = first_non_empty(
        product_schema.get("name") if product_schema else "",
        og_title.get("content") if og_title else "",
        h1.get_text(" ", strip=True) if h1 else "",
        title_tag.get_text(" ", strip=True) if title_tag else "",
    )

    description = extract_original_description(soup, product_schema)
    specs = extract_specs(soup)
    reviews = extract_reviews(soup, product_schema)
    image_urls = extract_images(soup, base_url, product_schema)
    page_excerpt = extract_page_excerpt(soup)

    return {
        "url": base_url,
        "title": title,
        "description": description,
        "specs": specs,
        "reviews": reviews,
        "image_urls": image_urls,
        "page_excerpt": page_excerpt,
    }


def fetch_image_base64(image_url: str, timeout_sec: float = 20.0) -> str:
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout_sec) as client:
            response = client.get(image_url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AppError(f"Nie udało się pobrać obrazu: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise AppError("URL grafiki nie zwrócił poprawnego pliku obrazu.")

    data = response.content
    if len(data) > 8 * 1024 * 1024:
        raise AppError("Grafika jest zbyt duża do analizy (limit 8 MB).")

    return base64.b64encode(data).decode("utf-8")


def ollama_chat(
    ollama_url: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout_sec: float = 150.0,
    as_json: bool = False,
) -> str:
    endpoint = ollama_url.rstrip("/") + "/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if as_json:
        payload["format"] = "json"

    try:
        response = httpx.post(endpoint, json=payload, timeout=timeout_sec)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AppError(f"Błąd połączenia z Ollama ({endpoint}): {exc}") from exc

    data = response.json()
    content = clean_text(data.get("message", {}).get("content", ""))
    if not content:
        raise AppError("Ollama zwróciła pustą odpowiedź.")
    return content


def parse_json_payload(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", value, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def analyze_visual(ollama_url: str, model: str, image_url: str) -> str:
    image_base64 = fetch_image_base64(image_url)

    messages = [
        {
            "role": "system",
            "content": (
                "Jesteś analitykiem e-commerce. Opisz obraz produktu rzeczowo, "
                "w języku polskim, bez wymyślania cech niewidocznych na zdjęciu."
            ),
        },
        {
            "role": "user",
            "content": (
                "Przeanalizuj grafikę produktu. Zwróć 4 krótkie punkty: "
                "1) co widać, 2) jakość ekspozycji, 3) emocja zakupowa, 4) ryzyka komunikacyjne."
            ),
            "images": [image_base64],
        },
    ]

    return ollama_chat(ollama_url, model, messages, timeout_sec=120.0, as_json=False)


def analyze_product(
    ollama_url: str,
    model: str,
    product_data: dict[str, Any],
    visual_notes: str,
) -> dict[str, Any]:
    prompt_payload = {
        "title": product_data.get("title", ""),
        "url": product_data.get("url", ""),
        "description": product_data.get("description", ""),
        "specs": product_data.get("specs", {}),
        "reviews": product_data.get("reviews", []),
        "page_excerpt": product_data.get("page_excerpt", ""),
        "visual_notes": visual_notes,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "Jesteś ekspertem e-commerce i copywriterem sprzedażowym. "
                "Twoje zadanie: poprawić opis produktu tak, aby zwiększać chęć zakupu, "
                "ale bez zmieniania faktów i bez obietnic bez pokrycia. "
                "Zwróć WYŁĄCZNIE poprawny JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Przeanalizuj dane produktu i wygeneruj wynik w JSON z polami: "
                "sentiment_summary (string), sentiment_score_0_100 (number), "
                "technical_summary (string), visual_summary (string), "
                "key_strengths (array[string]), key_objections (array[string]), "
                "original_description_short (string), proposed_description (string), "
                "why_better (array[string]), call_to_action (string). "
                "Proposed_description ma być po polsku, konkretny, perswazyjny, ale zgodny z faktami. "
                "Dane wejściowe:\n"
                f"{json.dumps(prompt_payload, ensure_ascii=False)}"
            ),
        },
    ]

    raw = ollama_chat(ollama_url, model, messages, timeout_sec=180.0, as_json=True)
    parsed = parse_json_payload(raw)
    if not parsed:
        raise AppError("Model zwrócił odpowiedź, ale nie w poprawnym JSON.")

    parsed.setdefault("sentiment_summary", "Brak danych")
    parsed.setdefault("sentiment_score_0_100", 50)
    parsed.setdefault("technical_summary", "Brak danych")
    parsed.setdefault("visual_summary", visual_notes)
    parsed.setdefault("key_strengths", [])
    parsed.setdefault("key_objections", [])
    parsed.setdefault(
        "original_description_short",
        truncate_text(product_data.get("description", "") or "Brak opisu źródłowego", 600),
    )
    parsed.setdefault("proposed_description", "")
    parsed.setdefault("why_better", [])
    parsed.setdefault("call_to_action", "")

    return parsed


def render_list(items: Any, empty_text: str = "Brak") -> None:
    if not isinstance(items, list) or not items:
        st.write(empty_text)
        return
    for item in items:
        st.write(f"- {clean_text(str(item))}")


def app() -> None:
    st.set_page_config(page_title="Generator opisu produktu", page_icon="🛍️", layout="wide")
    st.title("Generator atrakcyjnego opisu produktu")
    st.caption(
        "Wklej URL produktu. Aplikacja pobiera dane ze strony i używa Ollama, aby zaproponować "
        "bardziej sprzedażowy opis zgodny z faktami."
    )

    with st.sidebar:
        st.subheader("Ustawienia")
        ollama_url = st.text_input("URL Ollama", value=DEFAULT_OLLAMA_URL)
        model = st.text_input("Model tekstowy", value=DEFAULT_MODEL)
        vision_model = st.text_input("Model vision (opcjonalnie)", value="")
        st.caption("Jeśli używasz aliasu `gpt-20b`, wpisz go bezpośrednio w polu modelu.")

    product_url = st.text_input("URL produktu", placeholder="https://example.com/produkt")
    run = st.button("Analizuj produkt", type="primary")

    if not run:
        return

    if not is_valid_url(product_url):
        st.error("Podaj poprawny URL z prefiksem http:// lub https://")
        return

    with st.spinner("Pobieram stronę produktu..."):
        try:
            html, final_url = fetch_page_html(product_url)
            product_data = extract_product_data(html, final_url)
        except AppError as exc:
            st.error(str(exc))
            return

    st.subheader("Dane źródłowe")
    st.write(f"**Tytuł:** {product_data.get('title') or 'Brak'}")
    st.write(f"**URL po przekierowaniach:** {product_data.get('url')}")
    st.write(f"**Liczba wykrytych opinii:** {len(product_data.get('reviews', []))}")
    st.write(f"**Liczba wykrytych grafik:** {len(product_data.get('image_urls', []))}")

    with st.expander("Podgląd danych wejściowych do modelu"):
        st.json(
            {
                "title": product_data.get("title"),
                "description": truncate_text(product_data.get("description", ""), 900),
                "specs": product_data.get("specs", {}),
                "reviews": product_data.get("reviews", []),
                "image_urls": product_data.get("image_urls", []),
            }
        )

    visual_notes = "Brak analizy grafiki (nie znaleziono obrazu produktu)."
    image_urls = product_data.get("image_urls", [])
    if image_urls:
        image_url = image_urls[0]
        with st.spinner("Analizuję grafikę produktu przez Ollama..."):
            try:
                visual_notes = analyze_visual(
                    ollama_url=ollama_url,
                    model=vision_model.strip() or model.strip(),
                    image_url=image_url,
                )
            except AppError as exc:
                visual_notes = f"Analiza grafiki niedostępna: {exc}"

    with st.spinner("Analizuję produkt i tworzę nowy opis..."):
        try:
            analysis = analyze_product(
                ollama_url=ollama_url,
                model=model.strip(),
                product_data=product_data,
                visual_notes=visual_notes,
            )
        except AppError as exc:
            st.error(str(exc))
            return

    st.subheader("Porównanie opisu")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Stary opis")
        original = product_data.get("description") or analysis.get("original_description_short", "Brak")
        st.text_area("Opis źródłowy", value=original, height=260)

    with col_b:
        st.markdown("### Nowy opis (propozycja modelu)")
        st.text_area(
            "Opis zoptymalizowany",
            value=analysis.get("proposed_description", ""),
            height=260,
        )

    st.subheader("Wynik analizy")
    st.write(f"**Sentyment opinii:** {analysis.get('sentiment_summary', 'Brak')} ")
    st.write(f"**Ocena sentymentu (0-100):** {analysis.get('sentiment_score_0_100', 'Brak')}")
    st.write(f"**Podsumowanie techniczne:** {analysis.get('technical_summary', 'Brak')}")
    st.write(f"**Podsumowanie grafiki:** {analysis.get('visual_summary', visual_notes)}")

    st.markdown("### Mocne strony")
    render_list(analysis.get("key_strengths"))

    st.markdown("### Obiekcje zakupowe")
    render_list(analysis.get("key_objections"))

    st.markdown("### Dlaczego nowy opis jest lepszy")
    render_list(analysis.get("why_better"))

    cta = clean_text(str(analysis.get("call_to_action", "")))
    if cta:
        st.info(f"CTA: {cta}")


if __name__ == "__main__":
    app()
