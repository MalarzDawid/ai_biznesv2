# Generator atrakcyjnego opisu produktu

Aplikacja Streamlit przyjmuje URL produktu i:
- pobiera opis, parametry, opinie i grafiki ze strony,
- uruchamia analizę w Ollama (`gpt-oss:20b` lub alias `gpt-20b`),
- pokazuje porównanie: opis źródłowy vs nowy opis sprzedażowy.

## Wymagania

- Python 3.13+
- `uv`
- lokalnie uruchomiona Ollama (`http://localhost:11434`)

## Uruchomienie

```bash
uv sync
uv run streamlit run app.py
```

## Modele Ollama

Przykład przygotowania modelu:

```bash
ollama pull gpt-oss:20b
ollama serve
```

Jeśli używasz innego aliasu modelu (np. `gpt-20b`), wpisz go w panelu bocznym aplikacji.
