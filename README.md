# Amazon Sales Description Optimizer (v0.2)

Aplikacja Streamlit służąca do automatycznej optymalizacji kart produktów z serwisu Amazon za pomocą lokalnych modeli sztucznej inteligencji. Narzędzie pobiera dane produktu, analizuje recenzje klientów w celu zdefiniowania ich obiekcji, a następnie generuje perswazyjny opis marketingowy oraz propozycje nowej grafiki reklamowej.

## Funkcje w wersji 0.2

1. **Pobieranie i analizowanie danych:** Ekstrakcja tytułu, specyfikacji, oryginalnego opisu, opinii klientów oraz zdjęć produktu bezpośrednio z podanego adresu URL (Amazon).
2. **Optymalizacja opisu (LLM):** Generowanie nowego tekstu opartego na języku korzyści, dostosowanego do wybranego stylu wypowiedzi i zawierającego słowa kluczowe SEO.
3. **Personalizacja copywritingu:** 
   - **Styl (Tone of Voice):** Wybór między stylem Perswazyjnym (Sprzedażowym), Profesjonalnym (Eksperckim), Technicznym (Specyfikacyjnym) lub Luźnym (Młodzieżowym).
   - **SEO:** Wprowadzanie własnych słów kluczowych do wplecenia w tekst przez AI.
4. **Analiza wizualna i kreacja (FLUX.2):** Ocenianie oryginalnego zdjęcia za pomocą LLM i generowanie nowej, ulepszonej wersji grafiki produktowej na podstawie automatycznych promptów lifestyle/studio w lokalnym modelu FLUX.2.
5. **Uproszczony interfejs dla biznesu:** Nazewnictwo modeli oparte na czasie generowania (Szybki, Zbalansowany, Precyzyjny) oraz ukrycie skomplikowanych technicznych parametrów (CFG/Guidance, kroki inferencji).

## Wymagania

- Python 3.10+
- Menedżer pakietów `uv`
- Lokalnie uruchomiona Ollama (`http://localhost:11434`) z zainstalowanymi modelami:
  - `gemma3:4b` (Szybki)
  - `gpt-oss:20b` (Zbalansowany)
  - `gemma4:31b` (Precyzyjny)
- Opcjonalnie: Karta graficzna Nvidia (zalecane min. 24 GB VRAM) do uruchomienia lokalnego modelu generowania grafik FLUX.2-klein-4B.

## Uruchomienie

1. Zainstaluj zależności i zsynchronizuj środowisko:
   ```bash
   uv sync
   ```
2. Uruchom aplikację Streamlit:
   ```bash
   uv run streamlit run app.py
   ```

## Przygotowanie modeli w Ollama

Przed optymalizacją pobierz modele w terminalu:

```bash
ollama pull gemma3:4b
ollama pull gpt-oss:20b
ollama pull gemma4:31b
```
