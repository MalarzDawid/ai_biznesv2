from __future__ import annotations

import random
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

# List of realistic user agents to reduce blocking
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


class ScraperError(Exception):
    pass


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


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


def get_random_headers() -> dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }


def fetch_page_html(url: str, timeout_sec: float = 25.0) -> tuple[str, str]:
    headers = get_random_headers()
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout_sec) as client:
            response = client.get(url)
            # If we get a 503 or 403, we might be blocked by bot protection
            if response.status_code in (403, 503):
                raise ScraperError(
                    f"Strona zwróciła błąd HTTP {response.status_code}. Prawdopodobnie zostaliśmy zablokowani przez ochronę przed botami (np. Cloudflare/Amazon CAPTCHA)."
                )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ScraperError(f"Nie udało się pobrać strony: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        raise ScraperError("Podany URL nie wskazuje na stronę HTML produktu.")

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
            payload = json_loads_safe(raw)
            if payload:
                add_payload(payload)
        except Exception:
            continue

    return entries


def json_loads_safe(text: str) -> Any:
    import json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try cleaning up common JSON-LD formatting errors (like trailing commas)
        try:
            # Simple regex to remove trailing commas before closing braces/brackets
            cleaned = re.sub(r',\s*([\]}])', r'\1', text)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def find_product_schema(entries: list[dict[str, Any]]) -> dict[str, Any]:
    for item in entries:
        raw_type = item.get("@type")
        types = raw_type if isinstance(raw_type, list) else [raw_type]
        for value in types:
            if isinstance(value, str) and value.lower() == "product":
                return item
    return {}


def extract_title(soup: BeautifulSoup, product_schema: dict[str, Any]) -> str:
    # 1. Try schema
    schema_title = product_schema.get("name") if product_schema else ""
    
    # 2. Try Amazon specific title selectors
    amazon_title_node = soup.find(id="productTitle")
    amazon_title = amazon_title_node.get_text(strip=True) if amazon_title_node else ""
    
    # 3. Try standard H1
    h1_node = soup.find("h1")
    h1_title = h1_node.get_text(" ", strip=True) if h1_node else ""
    
    # 4. Try meta og:title
    og_title_node = soup.find("meta", property="og:title")
    og_title = og_title_node.get("content") if og_title_node else ""
    
    # 5. Try title tag
    title_tag = soup.find("title")
    title_tag_text = title_tag.get_text(" ", strip=True) if title_tag else ""

    return first_non_empty(schema_title, amazon_title, h1_title, og_title, title_tag_text)


def extract_specs(soup: BeautifulSoup) -> dict[str, str]:
    specs: dict[str, str] = {}

    # 1. Amazon specific technical details
    # Try looking for amazon table selector `#prodDetails` or `#technicalSpecifications_section_1`
    amazon_tables = soup.select("table.prodDetTable, #technicalSpecifications_section_1 table, #prodDetails table")
    for table in amazon_tables:
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) >= 2:
                key = clean_text(cells[0].get_text(" ", strip=True))
                val = clean_text(cells[1].get_text(" ", strip=True))
                # Remove extra noise from keys
                key = re.sub(r'\s+', ' ', key).strip()
                if key and val and key not in specs:
                    specs[key] = val
            if len(specs) >= 20:
                return specs

    # 2. Generic table rows
    if len(specs) < 5:
        for row in soup.select("table tr")[:150]:
            cells = row.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            key = clean_text(cells[0].get_text(" ", strip=True))
            val = clean_text(cells[1].get_text(" ", strip=True))
            if key and val and len(key) < 60 and len(val) < 250 and key not in specs:
                specs[key] = val
            if len(specs) >= 20:
                return specs

    # 3. DL description lists
    if len(specs) < 5:
        dts = soup.select("dl dt")[:80]
        for dt in dts:
            dd = dt.find_next_sibling("dd")
            if not dd:
                continue
            key = clean_text(dt.get_text(" ", strip=True))
            val = clean_text(dd.get_text(" ", strip=True))
            if key and val and len(key) < 60 and len(val) < 250 and key not in specs:
                specs[key] = val
            if len(specs) >= 20:
                break

    # 4. Bullet points formatted with colons
    if len(specs) < 5:
        for li in soup.select("li, .a-list-item")[:200]:
            text = clean_text(li.get_text(" ", strip=True))
            if ":" not in text:
                continue
            # Make sure it looks like a spec (Key: Value)
            parts = text.split(":", 1)
            key = clean_text(parts[0])
            val = clean_text(parts[1])
            if key and val and len(key) <= 50 and len(val) <= 200 and key not in specs:
                specs[key] = val
            if len(specs) >= 20:
                break

    return specs


def extract_reviews(soup: BeautifulSoup, product_schema: dict[str, Any]) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []

    # 1. Try JSON-LD reviews first
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
                        "author": clean_text(str(author)) or "klient",
                        "rating": rating,
                        "text": text,
                    }
                )

    # Add aggregate score if available
    agg = product_schema.get("aggregateRating")
    if isinstance(agg, dict):
        rating_value = clean_text(str(agg.get("ratingValue", "")))
        rating_count = clean_text(str(agg.get("reviewCount", "")))
        if rating_value or rating_count:
            collected.append(
                {
                    "author": "Średnia ocena sklepu",
                    "rating": rating_value,
                    "text": f"Średnia ocen produktu wynosi {rating_value} na podstawie {rating_count} opinii klientów.",
                }
            )

    # 2. Try Amazon specific review selectors (Customer Reviews section)
    amazon_reviews = soup.select('[data-hook="review"], .review, .a-section.review')
    for review in amazon_reviews[:20]:
        body_node = review.select_one(
            '[data-hook="reviewRichContentContainer"], '
            '[data-hook="review-body"], '
            '.review-text-content, '
            '.review-text, '
            '[data-hook="reviewText"]'
        )
        text = clean_text(body_node.get_text(" ", strip=True)) if body_node else ""
        
        # Clean up mobile or desktop teaser instructions if reviewText fallback was used
        if "Brief content visible" in text or "Full content visible" in text:
            text = text.replace("Brief content visible, double tap to read full content.", "")
            text = text.replace("Full content visible, double tap to read brief content.", "")
            text = clean_text(text)
            
        if not text:
            continue
        
        author_node = review.select_one('.a-profile-name, .author, [class*="author"]')
        author = clean_text(author_node.get_text(strip=True)) if author_node else "klient"
        
        rating_node = review.select_one('[data-hook="review-star-rating"], [class*="star-rating"], .a-icon-alt')
        rating = ""
        if rating_node:
            rating_text = rating_node.get_text(strip=True)
            rating_match = re.search(r"(\d+(\.\d+)?)", rating_text)
            if rating_match:
                rating = rating_match.group(1)

        collected.append({
            "author": author,
            "rating": rating,
            "text": text
        })

    # 3. Fallback generic selector for other platforms
    if len(collected) < 3:
        nodes = soup.select('[class*="review" i], [id*="review" i], [data-review]')
        for node in nodes[:35]:
            # Make sure it's not a giant section but an individual review
            text = clean_text(node.get_text(" ", strip=True))
            if 40 <= len(text) <= 800:
                collected.append({"author": "klient", "rating": "", "text": text})

    # Deduplicate based on review text
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in collected:
        key = item.get("text", "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= 15:  # Increased limit slightly to feed richer comments to LLM
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

    # 1. Try og:image
    meta_og = soup.find("meta", property="og:image")
    if meta_og and meta_og.get("content"):
        add_url(meta_og.get("content"))

    # 2. Try schema image
    schema_images = product_schema.get("image")
    if isinstance(schema_images, str):
        add_url(schema_images)
    elif isinstance(schema_images, list):
        for item in schema_images:
            if isinstance(item, str):
                add_url(item)
            elif isinstance(item, dict):
                add_url(item.get("url"))

    # 3. Try Amazon landing image
    amazon_img = soup.find(id="landingImage") or soup.find(id="imgBlkFront") or soup.select_one(".main-image-container img")
    if amazon_img:
        # Amazon uses 'data-old-hires' or 'data-a-dynamic-image'
        add_url(amazon_img.get("data-old-hires"))
        add_url(amazon_img.get("src"))
        dynamic_img = amazon_img.get("data-a-dynamic-image")
        if dynamic_img:
            # Parse dynamic image keys which are image URLs
            # format: {"url1":[width,height], "url2":[width,height]}
            try:
                import json
                parsed_dyn = json.loads(dynamic_img)
                for dyn_url in parsed_dyn.keys():
                    add_url(dyn_url)
            except Exception:
                pass

    # 4. Standard img elements (filtering out tiny elements like icons)
    for img in soup.select("img")[:60]:
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        # Exclude known tracking pixels, small icons
        if src and not any(x in src.lower() for x in ["icon", "logo", "sprite", "pixel", "loading", "spinner", "badge"]):
            add_url(src)

    return urls[:6]


def extract_original_description(soup: BeautifulSoup, product_schema: dict[str, Any]) -> str:
    # 1. Try Amazon Feature Bullets (about this item) - critical for Amazon products!
    bullets_node = soup.find(id="feature-bullets") or soup.find(id="featurebullets_feature_div")
    bullets_text = ""
    if bullets_node:
        lis = bullets_node.select("li:not(.a-list-item-invisible), .a-list-item")
        bullets_text = "\n".join([clean_text(li.get_text(strip=True)) for li in lis if clean_text(li.get_text(strip=True))])
    
    # 2. Try Amazon specific product description
    amazon_desc_node = soup.find(id="productDescription") or soup.select_one("#productDescription p")
    amazon_desc = clean_text(amazon_desc_node.get_text(" ", strip=True)) if amazon_desc_node else ""

    # 3. Try Schema description
    schema_desc = product_schema.get("description") if isinstance(product_schema, dict) else ""

    # 4. Try Meta descriptions
    meta_desc = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    og_desc = soup.find("meta", property="og:description")
    meta_desc_val = meta_desc.get("content") if meta_desc else ""
    og_desc_val = og_desc.get("content") if og_desc else ""

    desc = first_non_empty(
        amazon_desc,
        bullets_text,
        schema_desc,
        meta_desc_val,
        og_desc_val,
    )
    if desc:
        return truncate_text(desc, 3000)

    # 5. Fallback: paragraphs from main body
    paragraphs: list[str] = []
    for p in soup.select("article p, main p, p, #aplus p")[:50]:
        text = clean_text(p.get_text(" ", strip=True))
        # Filter out short or footer paragraphs
        if len(text) >= 60 and not any(x in text.lower() for x in ["cookies", "polityka prywatności", "regulamin", "prawa zastrzeżone"]):
            paragraphs.append(text)
        if len(paragraphs) >= 5:
            break

    return truncate_text("\n\n".join(paragraphs), 3000)


def extract_page_excerpt(soup: BeautifulSoup) -> str:
    chunks: list[str] = []
    for node in soup.select("h1, h2, h3, p, li, td")[:180]:
        text = clean_text(node.get_text(" ", strip=True))
        if len(text) < 20 or any(x in text.lower() for x in ["cookies", "polityka prywatności", "menu", "koszyk", "zaloguj"]):
            continue
        chunks.append(text)
        if sum(len(x) for x in chunks) >= 4500:
            break
    return truncate_text("\n".join(chunks), 4500)


def extract_product_data(html: str, base_url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    jsonld_entries = parse_json_ld(soup)
    product_schema = find_product_schema(jsonld_entries)

    title = extract_title(soup, product_schema)
    description = extract_original_description(soup, product_schema)
    reviews = extract_reviews(soup, product_schema)
    image_urls = extract_images(soup, base_url, product_schema)

    return {
        "url": base_url,
        "title": title,
        "description": description,
        "reviews": reviews,
        "image_urls": image_urls,
    }


def fetch_image_base64(image_url: str, timeout_sec: float = 20.0) -> str:
    import base64
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout_sec) as client:
            response = client.get(image_url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ScraperError(f"Nie udało się pobrać obrazu: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise ScraperError("URL grafiki nie zwrócił poprawnego pliku obrazu.")

    data = response.content
    if len(data) > 8 * 1024 * 1024:
        raise ScraperError("Grafika jest zbyt duża do analizy (limit 8 MB).")

    return base64.b64encode(data).decode("utf-8")
