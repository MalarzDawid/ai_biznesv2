from __future__ import annotations

import io
from PIL import Image
import streamlit as st
from src import scraper, llm

# Try to import image generation capabilities (optional dependency validation)
try:
    from src import image_gen
    HAS_IMAGE_GEN = True
    IMAGE_GEN_ERROR = ""
except Exception as e:
    HAS_IMAGE_GEN = False
    IMAGE_GEN_ERROR = str(e)

# Hardcoded local Ollama configurations
OLLAMA_URL = "http://localhost:11434"

def init_session_state() -> None:
    """Initialize state keys to persist results between user interactions."""
    if "product_url" not in st.session_state:
        st.session_state.product_url = ""
    if "original_description" not in st.session_state:
        st.session_state.original_description = ""
    if "optimized_description" not in st.session_state:
        st.session_state.optimized_description = ""
    if "product_title" not in st.session_state:
        st.session_state.product_title = ""
    if "reviews" not in st.session_state:
        st.session_state.reviews = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemma4:31b"
    if "image_urls" not in st.session_state:
        st.session_state.image_urls = []
    if "generated_image" not in st.session_state:
        st.session_state.generated_image = None
    if "image_prompts" not in st.session_state:
        st.session_state.image_prompts = []
    if "image_description" not in st.session_state:
        st.session_state.image_description = ""
    if "custom_image_prompt" not in st.session_state:
        st.session_state.custom_image_prompt = ""
    if "selected_concept" not in st.session_state:
        st.session_state.selected_concept = 0
    if "selected_tone" not in st.session_state:
        st.session_state.selected_tone = "Perswazyjny (Sprzedażowy)"
    if "seo_keywords" not in st.session_state:
        st.session_state.seo_keywords = ""

def count_sentences(text: str) -> int:
    import re
    # Simple split based on punctuation (. ! ?) followed by whitespace or end of string
    sentences = re.split(r'[.!?]+(?:\s+|$)', text.strip())
    sentences = [s for s in sentences if s.strip()]
    return max(1, len(sentences))

def parse_ollama_output(output: str) -> tuple[str, str, list[str]]:
    import re
    """
    Parses Ollama output to extract:
    1. Polish sales description
    2. English description of the main image
    3. List of 3 English image prompt proposals
    """
    # Extract OPIS
    opis_match = re.search(r"\[OPIS\](.*?)\[/OPIS\]", output, re.DOTALL | re.IGNORECASE)
    if opis_match:
        opis = opis_match.group(1).strip()
    else:
        # Fallback if tags not found: take everything before any prompt tag
        parts = re.split(r"\[OPIS_OBRAZU\]|\[PROMPT_1\]", output, flags=re.IGNORECASE)
        opis = parts[0].replace("[OPIS]", "").replace("[/OPIS]", "").strip()

    # Extract OPIS_OBRAZU
    img_desc_match = re.search(r"\[OPIS_OBRAZU\](.*?)\[/OPIS_OBRAZU\]", output, re.DOTALL | re.IGNORECASE)
    if img_desc_match:
        img_desc = img_desc_match.group(1).strip()
    else:
        # Fallback search if tags missing
        img_desc = "The main product image shows the product centered on a clean, solid white studio background."

    # Extract PROMPTs
    prompts = []
    for tag in ["PROMPT_1", "PROMPT_2", "PROMPT_3"]:
        prompt_match = re.search(rf"\[{tag}\](.*?)\[/{tag}\]", output, re.DOTALL | re.IGNORECASE)
        if prompt_match:
            prompts.append(prompt_match.group(1).strip())
        else:
            # Fallback regex search
            prompt_match_no_close = re.search(rf"\[{tag}\](.*?)(?=\[|$)", output, re.DOTALL | re.IGNORECASE)
            if prompt_match_no_close:
                prompts.append(prompt_match_no_close.group(1).strip())

    # Fallback default prompts if we don't have enough
    while len(prompts) < 3:
        i = len(prompts) + 1
        if i == 1:
            prompts.append("high quality professional studio product photography, clean studio background, soft lighting, 8k resolution, commercial advertising")
        elif i == 2:
            prompts.append("lifestyle product photography, in-use setting, natural soft lighting, warm tones, high detail, commercial shot")
        else:
            prompts.append("creative advertising photography, dramatic dynamic lighting, colorful abstract background, sharp focus, 8k resolution")

    return opis, img_desc, prompts[:3]



def main() -> None:
    st.set_page_config(
        page_title="Amazon Sales Description Optimizer",
        page_icon="🛍️",
        layout="wide"
    )

    # Styling for premium Google-like visual appearance
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Hide Streamlit headers and footers to look like custom app */
        header {visibility: hidden !important; height: 0px !important;}
        footer {visibility: hidden !important; height: 0px !important;}
        #MainMenu {visibility: hidden !important;}

        /* CSS Variables for Auto Light/Dark Themes */
        :root {
            --google-blue: #1a73e8;
            --google-blue-hover: #1557b0;
            --google-blue-light: #e8f0fe;
            --bg-page: #f8f9fa;
            --bg-card: #ffffff;
            --border-color: #dadce0;
            --text-primary: #202124;
            --text-secondary: #5f6368;
            --input-bg: #ffffff;
            --card-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 2px 6px 2px rgba(60,64,67,0.15);
            --subtle-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --google-blue: #8ab4f8;
                --google-blue-hover: #aecbfa;
                --google-blue-light: rgba(138, 180, 248, 0.15);
                --bg-page: #121212;
                --bg-card: #1e1e1e;
                --border-color: #3c4043;
                --text-primary: #e8eaed;
                --text-secondary: #9aa0a6;
                --input-bg: #202124;
                --card-shadow: 0 1px 3px 0 rgba(0,0,0,0.5), 0 4px 8px 3px rgba(0,0,0,0.3);
                --subtle-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }
        }

        /* Base Body and Layout adjustments */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        }

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            max-width: 1280px !important;
        }

        /* Custom Google-like Top Navigation Banner */
        .google-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 28px;
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            margin-bottom: 28px;
            box-shadow: var(--subtle-shadow);
        }
        .google-nav-brand {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .brand-accent {
            font-size: 1.4rem;
            color: var(--google-blue);
            font-weight: bold;
            display: inline-block;
            animation: rotate 6s linear infinite;
        }
        @keyframes rotate {
            100% { transform: rotate(360deg); }
        }
        .product-name {
            font-weight: 700;
            font-size: 1.25rem;
            color: var(--text-primary);
            letter-spacing: -0.4px;
        }
        .google-nav-badge {
            background-color: var(--google-blue-light);
            color: var(--google-blue);
            padding: 6px 16px;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.3px;
            border: 1px solid rgba(26, 115, 232, 0.15);
        }

        /* Inputs & Elements Styling */
        div[data-testid="stTextInput"] input, 
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {
            background-color: var(--input-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 10px 14px !important;
            font-size: 0.95rem !important;
            transition: all 0.25s ease !important;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.02) !important;
        }

        div[data-testid="stTextInput"] input:focus, 
        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus {
            border-color: var(--google-blue) !important;
            box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.15) !important;
        }

        /* Form Expander overrides */
        div[data-testid="stExpander"] {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 16px !important;
            box-shadow: var(--subtle-shadow) !important;
            margin-bottom: 20px !important;
            overflow: hidden !important;
            transition: box-shadow 0.25s ease !important;
        }
        div[data-testid="stExpander"]:hover {
            box-shadow: 0 8px 24px rgba(0,0,0,0.06) !important;
        }
        div[data-testid="stExpander"] summary {
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            padding: 16px 20px !important;
            font-size: 1.05rem !important;
        }

        /* Widget labels styling */
        div[data-testid="stWidgetLabel"] p {
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            color: var(--text-primary) !important;
            margin-bottom: 6px !important;
        }

        /* Premium Buttons Styling */
        div.stButton > button {
            background: var(--google-blue) !important;
            color: var(--bg-card) !important;
            border: 1px solid transparent !important;
            border-radius: 28px !important;
            padding: 12px 28px !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.15px !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06) !important;
        }

        div.stButton > button:hover {
            background: var(--google-blue-hover) !important;
            box-shadow: var(--card-shadow) !important;
            transform: translateY(-1px) !important;
        }

        div.stButton > button:active {
            transform: translateY(1px) !important;
        }

        /* Tabs UI overrides */
        div[role="tablist"] {
            border-bottom: 1px solid var(--border-color) !important;
            margin-bottom: 24px !important;
            gap: 8px !important;
        }
        button[role="tab"] {
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            color: var(--text-secondary) !important;
            border: none !important;
            background: transparent !important;
            padding: 10px 20px !important;
            border-radius: 20px 20px 0 0 !important;
            transition: all 0.2s ease !important;
        }
        button[role="tab"]:hover {
            color: var(--google-blue) !important;
            background-color: var(--google-blue-light) !important;
        }
        button[role="tab"][aria-selected="true"] {
            color: var(--google-blue) !important;
            border-bottom: 3px solid var(--google-blue) !important;
        }

        /* Download buttons and secondary button details */
        .stDownloadButton > button {
            background-color: transparent !important;
            color: var(--google-blue) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 24px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            transition: all 0.2s ease !important;
            box-shadow: none !important;
        }
        .stDownloadButton > button:hover {
            background-color: var(--google-blue-light) !important;
            border-color: var(--google-blue) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Default values to prevent UnboundLocalError
    img_guidance = 4.0
    img_steps = 30

    init_session_state()

    # Render beautiful Google-style Top Navigation Banner
    st.markdown(
        """
        <div class="google-nav">
            <div class="google-nav-brand">
                <span class="brand-accent">✦</span>
                <span class="product-name">RetailOptima AI &nbsp;<span style="font-weight: 300; font-size: 0.95rem; opacity: 0.6;">| &nbsp; PDP Optimizer</span></span>
            </div>
            <div class="google-nav-badge">Enterprise Suite</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dual Input layout: URL Input and Model Dropdown
    col_input1, col_input2 = st.columns([3, 1])
    with col_input1:
        product_url = st.text_input(
            "Adres URL produktu z Amazon",
            value=st.session_state.product_url,
            placeholder="Wklej link, np. https://www.amazon.pl/dp/B0..."
        )
    with col_input2:
        model_names = {
            "gemma3:4b": "Szybki (gemma3:4b) - ok. 15-30 sek",
            "gpt-oss:20b": "Zbalansowany (gpt-oss:20b) - ok. 1-2 min",
            "gemma4:31b": "Precyzyjny (gemma4:31b) - ok. 3-5 min"
        }
        st.selectbox(
            "Model językowy (Ollama)",
            options=["gemma3:4b", "gpt-oss:20b", "gemma4:31b"],
            key="selected_model",
            format_func=lambda x: model_names.get(x, x),
            help="• Szybki: model 4B (bardzo szybki czas generowania, uproszczona jakość)\n\n"
                 "• Zbalansowany: model 20B (optymalna równowaga między czasem a jakością)\n\n"
                 "• Precyzyjny: model 31B (najwyższa jakość, bogate słownictwo sprzedażowe, ale najdłuższy czas oczekiwania)"
        )

    # Słownik stylów copywritingu dla modelu językowego
    tone_options = {
        "Perswazyjny (Sprzedażowy)": "Skupia się na wywoływaniu emocji zakupowych, podkreślaniu korzyści (język korzyści) i silnym wezwaniu do działania (CTA).",
        "Profesjonalny (Ekspercki)": "Styl poważny, merytoryczny, budujący autorytet i zaufanie do marki. Idealny dla produktów premium, biznesowych (B2B) i droższej elektroniki.",
        "Techniczny (Specyfikacyjny)": "Koncentruje się na szczegółach technicznych, parametrach, faktach i precyzyjnym opisie konstrukcji lub działania produktu.",
        "Luźny (Młodzieżowy)": "Bezpośredni i przyjazny ton (zwracanie się do odbiorcy na 'Ty'), pełen energii i entuzjazmu, idealny dla marek lifestylowych i młodego grona odbiorców."
    }

    # Sekcja personalizacji opisu (styl + SEO)
    with st.expander("✍️ Personalizacja stylu copywritingu i słowa kluczowe (SEO)", expanded=True):
        col_pers1, col_pers2 = st.columns(2)
        with col_pers1:
            st.selectbox(
                "Styl i ton wypowiedzi (Tone of Voice)",
                options=list(tone_options.keys()),
                key="selected_tone",
                help="Wybierz styl, w jakim sztuczna inteligencja ma napisać nowy opis produktu."
            )
            st.caption(f"ℹ️ **O wybranym stylu:** {tone_options[st.session_state.selected_tone]}")
        with col_pers2:
            st.text_input(
                "Słowa kluczowe pod SEO (rozdziel przecinkami)",
                key="seo_keywords",
                placeholder="np. słuchawki bezprzewodowe, redukcja szumów ANC, bluetooth 5.3",
                help="Wpisz frazy kluczowe, które sztuczna inteligencja ma naturalnie wpleść w treść opisu w celu poprawienia widoczności w wyszukiwarkach."
            )

    # Usunięto zaawansowane ustawienia FLUX dla wygody użytkownika biznesowego.
    # W tle używane są domyślne parametry (Siła promptu = 4.0, Kroki = 30).

    # Clear state if URL changes
    if product_url != st.session_state.product_url:
        st.session_state.product_url = product_url
        st.session_state.original_description = ""
        st.session_state.optimized_description = ""
        st.session_state.product_title = ""
        st.session_state.reviews = []
        st.session_state.image_urls = []
        st.session_state.generated_image = None
        st.session_state.image_prompts = []
        st.session_state.image_description = ""
        st.session_state.custom_image_prompt = ""
        if "selected_concept" in st.session_state:
            st.session_state.selected_concept = 0

    # Submit button
    run_opt = st.button("🚀 Pobierz i optymalizuj opis", type="primary")

    if run_opt:
        if not product_url.strip():
            st.error("Wprowadź poprawny adres URL produktu z serwisu Amazon.")
            return

        # Clear FLUX pipeline from VRAM to make room for Ollama (gemma4:31b is very large)
        if HAS_IMAGE_GEN:
            image_gen.clear_gpu_cache()

        # Clear previous state
        st.session_state.product_title = ""
        st.session_state.original_description = ""
        st.session_state.optimized_description = ""
        st.session_state.reviews = []
        st.session_state.image_urls = []
        st.session_state.generated_image = None
        st.session_state.image_prompts = []
        st.session_state.image_description = ""
        st.session_state.custom_image_prompt = ""
        if "selected_concept" in st.session_state:
            st.session_state.selected_concept = 0

        with st.status("Przetwarzanie karty produktu...", expanded=True) as status:
            try:
                status.write("Pobieranie i analizowanie kodu strony Amazon...")
                html, final_url = scraper.fetch_page_html(product_url)
                product_data = scraper.extract_product_data(html, final_url)
                
                title = product_data.get("title", "")
                desc = product_data.get("description", "")
                reviews = product_data.get("reviews", [])
                image_urls = product_data.get("image_urls", [])
                
                st.session_state.product_title = title
                st.session_state.original_description = desc
                st.session_state.reviews = reviews
                st.session_state.image_urls = image_urls
                
                status.write(f"✓ Pomyślnie pobrano: **{title}**")
                
            except Exception as e:
                status.update(label="❌ Błąd pobierania danych", state="error")
                st.error(f"Nie udało się pobrać opisu ze strony: {e}")
                return

            if not st.session_state.original_description:
                status.update(label="❌ Brak opisu", state="error")
                st.error("Nie znaleziono opisu na podanej stronie produktu.")
                return

            # STEP 2: Optimize using selected model via Ollama
            status.write(f"Generowanie zoptymalizowanego opisu za pomocą {st.session_state.selected_model}...")
            status.update(label="🤖 Trwa optymalizacja opisu przez model AI...", state="running")
            
            num_orig_sentences = count_sentences(st.session_state.original_description)
            max_new_sentences = num_orig_sentences + 2

            # Format reviews for the LLM prompt context
            formatted_reviews = ""
            if st.session_state.reviews:
                formatted_reviews = "\nOpinie klientów o produkcie:\n" + "\n".join([f"- {r.get('text')}" for r in st.session_state.reviews[:8]])

            # Custom copywriting tone and SEO instructions
            tone_instr = f"- Ton wypowiedzi: Użyj stylu '{st.session_state.selected_tone}'. Charakterystyka tego stylu: {tone_options.get(st.session_state.selected_tone, '')}"
            seo_instr = ""
            if st.session_state.seo_keywords.strip():
                seo_instr = f"- Słowa kluczowe SEO: Wpleć w treść w naturalny i płynny sposób następujące frazy kluczowe (zadbaj o poprawne gramatycznie wbudowanie ich w polski tekst): {st.session_state.seo_keywords.strip()}."

            prompt_content = f"""Jesteś doświadczonym ekspertem e-commerce i copywriterem. Twoim zadaniem jest:
1. Zoptymalizowanie opisu produktu pod kątem zwiększenia sprzedaży (conversion rate) w języku polskim.
2. Stworzenie szczegółowego opisu głównego zdjęcia produktu po angielsku (na podstawie załączonego obrazu lub informacji o produkcie).
3. Zaproponowanie 3 różnych, kreatywnych i ciekawych nowych promptów (propozycji) po angielsku do wygenerowania nowego obrazu głównego za pomocą modelu FLUX.2. Te propozycje powinny być ciekawsze niż oryginalny obraz (np. pokazujące produkt w luksusowym studiu, w użyciu/lifestyle, w dynamicznej aranżacji).

Tytuł produktu: {title}

Oryginalny opis:
{st.session_state.original_description}
{formatted_reviews}

WYMAGANIA DOTYCZĄCE OPISU MARKETINGOWEGO (W JĘZYKU POLSKIM):
- Używaj języka korzyści (cecha -> zaleta -> korzyść).
- Zachowaj 100% zgodności z faktami i specyfikacją.
- Wykorzystaj opinie klientów (podkreśl zalety, odeprzyj obiekcje).
- Dodaj przejrzyste nagłówki, akapity i listę wypunktowaną oraz CTA na końcu.
{tone_instr}
{seo_instr}
- Zwięzłość: Oryginalny opis składa się z {num_orig_sentences} zdań. Nowy opis MUSI liczyć maksymalnie {max_new_sentences} zdań (nie może być dłuższy niż oryginalny o więcej niż 2 zdania).

WYMAGANIA DOTYCZĄCE PROPOZYCJI OBRAZÓW (W JĘZYKU ANGIELSKIM):
- Powinny być sformułowane jako szczegółowe prompty dla modelu generowania obrazów FLUX (np. "lifestyle product photography of..., clean studio background, soft lighting, 8k...").
- Każdy z 3 promptów powinien proponować inną, ciekawą, estetyczną kompozycję i otoczenie produktu, aby obraz był bardziej atrakcyjny i dynamiczny niż standardowe, nudne tło.

FORMAT REZULTATU (Zastosuj dokładnie te znaczniki/tagi, nie pisz żadnych dodatkowych wstępów ani komentarzy):

[OPIS]
(Tutaj umieść zoptymalizowany opis produktu w języku polskim)
[/OPIS]

[OPIS_OBRAZU]
(Tutaj umieść szczegółowy opis oryginalnego obrazu głównego w języku angielskim)
[/OPIS_OBRAZU]

[PROMPT_1]
(Tutaj umieść pierwszą propozycję nowego obrazu głównego w języku angielskim)
[/PROMPT_1]

[PROMPT_2]
(Tutaj umieść drugą propozycję nowego obrazu głównego w języku angielskim)
[/PROMPT_2]

[PROMPT_3]
(Tutaj umieść trzecią propozycję nowego obrazu głównego w języku angielskim)
[/PROMPT_3]"""

            messages = [
                {"role": "user", "content": prompt_content}
            ]
            
            # Place static placeholder inside status container to stream the response live
            stream_placeholder = st.empty()
            accumulated_text = ""
            
            try:
                # Try passing the image if it exists to support multimodal vision models in Ollama
                if st.session_state.image_urls:
                    try:
                        status.write("Pobieranie zdjęcia głównego do analizy wizualnej przez model AI...")
                        img_base64 = scraper.fetch_image_base64(st.session_state.image_urls[0])
                        messages[0]["images"] = [img_base64]
                    except Exception as img_exc:
                        status.write(f"⚠️ Nie udało się pobrać zdjęcia do analizy (pomijanie obrazu): {img_exc}")

                status.write(f"Generowanie zoptymalizowanego opisu oraz propozycji grafik za pomocą {st.session_state.selected_model}...")
                status.update(label="🤖 Trwa generowanie przez model AI...", state="running")
                
                try:
                    for chunk in llm.stream_chat(
                        provider="ollama",
                        model=st.session_state.selected_model,
                        messages=messages,
                        ollama_url=OLLAMA_URL
                    ):
                        accumulated_text += chunk
                        stream_placeholder.text_area("Generowanie na żywo...", value=accumulated_text, height=300, disabled=True)
                except Exception as ollama_exc:
                    # If we passed an image and it failed, let's retry WITHOUT the image!
                    if "images" in messages[0]:
                        status.write("⚠️ Model prawdopodobnie nie obsługuje obrazów. Ponawianie próby w trybie tekstowym...")
                        del messages[0]["images"]
                        accumulated_text = ""
                        for chunk in llm.stream_chat(
                            provider="ollama",
                            model=st.session_state.selected_model,
                            messages=messages,
                            ollama_url=OLLAMA_URL
                        ):
                            accumulated_text += chunk
                            stream_placeholder.text_area("Generowanie na żywo (tylko tekst)...", value=accumulated_text, height=300, disabled=True)
                    else:
                        raise ollama_exc

                # Parse the response
                opis, img_desc, prompts = parse_ollama_output(accumulated_text)
                st.session_state.optimized_description = opis
                st.session_state.image_description = img_desc
                st.session_state.image_prompts = prompts
                
                # Pre-populate custom prompt with the first option
                if prompts:
                    st.session_state.custom_image_prompt = prompts[0]
                    st.session_state.selected_concept = 0

                status.write("Zwalnianie pamięci GPU (wyładowywanie modelu Ollama)...")
                llm.unload_model(st.session_state.selected_model, ollama_url=OLLAMA_URL)
                status.update(label="✓ Ukończono optymalizację!", state="complete")
                stream_placeholder.empty()
                st.rerun()
            except Exception as e:
                status.update(label="❌ Błąd optymalizacji AI", state="error")
                st.error(f"Wystąpił błąd podczas komunikacji z Ollama (upewnij się, że Ollama działa i model {st.session_state.selected_model} jest zainstalowany): {e}")
                return

    # Render results using tabs if description is generated
    if st.session_state.original_description:
        st.markdown("---")
        if st.session_state.product_title:
            st.markdown(f"## **{st.session_state.product_title}**")
            
        tab_text, tab_image = st.tabs(["📝 Zoptymalizowany opis", "🎨 Grafika (FLUX)"])
        
        with tab_text:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔴 Oryginalny opis z Amazon")
                st.text_area(
                    "Oryginalna treść",
                    value=st.session_state.original_description,
                    height=500,
                    key="orig_desc_display",
                    disabled=True
                )
                if st.session_state.reviews:
                    with st.expander(f"💬 Pobrane opinie klientów ({len(st.session_state.reviews)})"):
                        for r in st.session_state.reviews:
                            rating_str = f" ⭐ {r.get('rating')}" if r.get('rating') else ""
                            st.markdown(f"**Klient{rating_str}**")
                            st.caption(r.get("text", ""))
                            st.markdown("---")
            with col2:
                st.subheader("🟢 Zoptymalizowany opis sprzedażowy")
                st.text_area(
                    "Nowa ulepszona treść",
                    value=st.session_state.optimized_description,
                    height=500,
                    key="opt_desc_display",
                    disabled=True
                )
                if st.session_state.optimized_description:
                    st.download_button(
                        label="💾 Pobierz zoptymalizowany opis (.txt)",
                        data=st.session_state.optimized_description,
                        file_name="zoptymalizowany_opis.txt",
                        mime="text/plain",
                        key="dl_desc_btn"
                    )
                    
        with tab_image:
            col_img, col_ctrl = st.columns([1, 1])
            with col_img:
                st.subheader("🖼️ Zdjęcia produktu")
                if st.session_state.image_urls:
                    st.image(st.session_state.image_urls[0], caption="Oryginalny obraz produktu", use_container_width=True)
                if st.session_state.generated_image:
                    st.markdown("---")
                    st.image(st.session_state.generated_image, caption="Ulepszona grafika (FLUX)", use_container_width=True)
                    
                    # Download button for the enhanced image
                    img_byte_arr = io.BytesIO()
                    st.session_state.generated_image.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    st.download_button(
                        label="💾 Pobierz ulepszony obraz (PNG)",
                        data=img_bytes,
                        file_name="ulepszona_grafika.png",
                        mime="image/png",
                        key="dl_img_btn",
                        use_container_width=True
                    )
            with col_ctrl:
                st.subheader("⚙️ Panel generowania obrazu")
                
                # Show English description of the main image if available
                if st.session_state.image_description:
                    with st.expander("📝 Oryginalny opis obrazu (po angielsku)", expanded=True):
                        st.write(st.session_state.image_description)
                
                if st.session_state.image_prompts:
                    st.markdown("### Wybierz kreatywną koncepcję dla FLUX:")
                    
                    # Initialize session states for concept selection if not present
                    if "selected_concept" not in st.session_state:
                        st.session_state.selected_concept = 0
                        st.session_state.custom_image_prompt = st.session_state.image_prompts[0]
                        
                    def update_prompt():
                        st.session_state.custom_image_prompt = st.session_state.image_prompts[st.session_state.selected_concept]
                        
                    st.radio(
                        "Wybierz koncepcję:",
                        options=[0, 1, 2],
                        format_func=lambda x: f"Koncepcja {x+1}: {st.session_state.image_prompts[x]}",
                        key="selected_concept",
                        on_change=update_prompt
                    )
                else:
                    st.info("Brak gotowych koncepcji z modelu AI. Wpisz własny prompt poniżej.")
                
                # Prompt text area
                if "custom_image_prompt" not in st.session_state:
                    st.session_state.custom_image_prompt = ""
                    
                edited_prompt = st.text_area(
                    "Dostosuj prompt dla FLUX (po angielsku)",
                    value=st.session_state.custom_image_prompt,
                    key="custom_image_prompt_textarea",
                    height=150
                )
                st.session_state.custom_image_prompt = edited_prompt

                # Action button to generate
                if HAS_IMAGE_GEN:
                    st.markdown("---")
                    if st.button("🎨 Wygeneruj ulepszoną grafikę (FLUX)", key="gen_img_btn", type="primary"):
                        with st.status("🎨 Generowanie ulepszonej grafiki...", expanded=True) as img_status:
                            try:
                                import httpx
                                # Free up VRAM by unloading the Ollama model before loading FLUX
                                img_status.write("Zwalnianie pamięci GPU (wyładowywanie modelu Ollama)...")
                                llm.unload_model(st.session_state.selected_model, ollama_url=OLLAMA_URL)
                                
                                main_img_url = st.session_state.image_urls[0]
                                img_status.write("Pobieranie oryginalnego zdjęcia głównego...")
                                img_response = httpx.get(main_img_url, timeout=15.0)
                                if img_response.status_code == 200:
                                    init_img = Image.open(io.BytesIO(img_response.content))
                                    
                                    img_status.write("Uruchamianie lokalnego modelu FLUX.2-klein-4B na RTX 3090...")
                                    
                                    # Set prompt to the edited text area prompt, or fallback
                                    final_prompt = st.session_state.custom_image_prompt.strip()
                                    if not final_prompt:
                                        short_title = st.session_state.product_title.split(',')[0].split('-')[0].strip()
                                        final_prompt = f"high quality professional studio product photography of {short_title}, clean studio background, soft lighting, 8k resolution, highly detailed, commercial advertising, sharp focus"
                                    
                                    # Render beautiful pulsing loading card
                                    anim_placeholder = st.empty()
                                    animation_html = """
                                    <style>
                                    @keyframes gemini-gradient {
                                      0% { background-position: 0% 50%; }
                                      50% { background-position: 100% 50%; }
                                      100% { background-position: 0% 50%; }
                                    }
                                    .gemini-loader-card {
                                      width: 100%;
                                      height: 380px;
                                      background: linear-gradient(-45deg, #4285F4, #9B51E0, #E94235, #F2994A);
                                      background-size: 400% 400%;
                                      animation: gemini-gradient 10s ease infinite;
                                      border-radius: 16px;
                                      display: flex;
                                      flex-direction: column;
                                      align-items: center;
                                      justify-content: center;
                                      color: white;
                                      box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                                      text-align: center;
                                      padding: 24px;
                                    }
                                    .gemini-sparkle {
                                      font-size: 3.5rem;
                                      margin-bottom: 16px;
                                      animation: float 3s ease-in-out infinite;
                                    }
                                    @keyframes float {
                                      0% { transform: translateY(0px) rotate(0deg); }
                                      50% { transform: translateY(-10px) rotate(5deg); }
                                      100% { transform: translateY(0px) rotate(0deg); }
                                    }
                                    .gemini-text {
                                      font-size: 1.3rem;
                                      font-weight: 600;
                                      letter-spacing: 0.5px;
                                      margin-bottom: 8px;
                                      text-shadow: 0 2px 4px rgba(0,0,0,0.2);
                                    }
                                    .gemini-subtext {
                                      font-size: 0.95rem;
                                      opacity: 0.85;
                                      text-shadow: 0 1px 2px rgba(0,0,0,0.2);
                                    }
                                    </style>
                                    <div class="gemini-loader-card">
                                      <div class="gemini-sparkle">✦</div>
                                      <div class="gemini-text">RetailOptima AI Engine</div>
                                      <div class="gemini-subtext">Trwa generowanie ulepszonej grafiki reklamowej przez model FLUX...</div>
                                    </div>
                                    """
                                    anim_placeholder.markdown(animation_html, unsafe_allow_html=True)
                                    
                                    progress_placeholder = st.empty()
                                    
                                    # Callback function to update the progress bar step-by-step
                                    def flux_callback(pipe, step_index, timestep, callback_kwargs):
                                        progress = (step_index + 1) / img_steps
                                        progress_placeholder.progress(
                                            progress,
                                            text=f"Generowanie grafiki: krok {step_index + 1}/{img_steps}"
                                        )
                                        return callback_kwargs
                                    
                                    generated_img = image_gen.generate_img2img(
                                        model_id="black-forest-labs/FLUX.2-klein-4B",
                                        init_image=init_img,
                                        prompt=final_prompt,
                                        guidance_scale=img_guidance,
                                        num_inference_steps=img_steps,
                                        seed=-1,
                                        callback=flux_callback
                                    )
                                    st.session_state.generated_image = generated_img
                                    anim_placeholder.empty()
                                    progress_placeholder.empty()
                                    
                                    img_status.write("✓ Pomyślnie wygenerowano ulepszoną grafikę!")
                                    img_status.update(label="✓ Ukończono!", state="complete")
                                    st.rerun()
                                else:
                                    img_status.write(f"⚠️ Nie udało się pobrać zdjęcia głównego: status {img_response.status_code}")
                            except Exception as e:
                                img_status.write(f"⚠️ Nie udało się wygenerować grafiki: {e}")
                else:
                    st.markdown("---")
                    st.warning("⚠️ Funkcja generowania i ulepszania grafik (FLUX) jest niedostępna (brak bibliotek lub odpowiedniej karty graficznej).")

if __name__ == "__main__":
    main()
