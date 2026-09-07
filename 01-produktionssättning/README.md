# Från notebook till körbar tjänst

Exempel från lektionen 7/9. Samma struktur förväntas i projektets applikationsdel.

Projektet hanteras med [uv](https://docs.astral.sh/uv/): en `pyproject.toml`
beskriver paketet och dess beroenden, `uv.lock` låser exakta versioner och
`uv run` kör kommandon i en automatiskt skapad virtuell miljö.

## Installera uv (en gång per dator)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv --version                     # kontrollera att det fungerar (öppna ny terminal först)
```

## Kom igång (från en ren klon)

```bash
uv sync                                          # 0. skapar .venv och installerar allt (inkl. dev-beroenden)

uv run train                                     # 1. tränar och sparar model/iris_pipeline.joblib
uv run uvicorn iris_service.api:app --reload     # 2. startar API:et -> http://127.0.0.1:8000/docs
uv run pytest -v                                 # 3. kör testerna
uv run streamlit run src/iris_service/ui.py      # 4. alternativt gränssnitt, samma modell
```

`uv sync` laddar själv ner rätt Python-version (se `.python-version`) om den saknas.
Ingen `source .venv/bin/activate` behövs: `uv run` aktiverar miljön åt dig.

`uv run train` fungerar för att `pyproject.toml` deklarerar ett kommando under
`[project.scripts]` som pekar på funktionen `main` i `train.py`. Vill ni ha fler
kommandon (t.ex. `serve`) är det där de läggs till.

### Testa API:et manuellt

När API:et är igång: öppna <http://127.0.0.1:8000/docs> och prova `/predict` i Swagger UI,
eller anropa det från terminalen:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Förväntat svar: `{"species": "setosa", "class_index": 0, "probabilities": {...}}`.
Skicka in ett negativt värde eller utelämna ett fält och ni får HTTP 422 tillbaka.

## Struktur

```
01-produktionssättning/
├── pyproject.toml          # Paketnamn, beroenden (pinnade!), kommandon
├── uv.lock                 # Exakta versioner av ALLA paket, även indirekta
├── .python-version         # Python-version som uv använder
├── .gitignore              # .venv/, model/, cache: sådant som inte hör hemma i git
├── src/
│   └── iris_service/       # Själva paketet
│       ├── __init__.py     # Gör mappen till ett importerbart paket
│       ├── config.py       # Sökvägar, utgår från projektroten
│       ├── train.py        # Steg 1: tränar Pipeline (StandardScaler + LogisticRegression), sparar joblib + metadata
│       ├── api.py          # Steg 2: FastAPI-app, laddar modellen vid uppstart, /health och /predict
│       └── ui.py           # Steg 4: Streamlit-frontend som använder samma joblib-fil
├── tests/
│   └── test_api.py         # Steg 3: tester med TestClient
├── model/                  # Tränad modell + metadata (skapas av train, ligger inte i git)
└── .venv/                  # Virtuell miljö (skapas av uv sync, ligger inte i git)
```

Varför `src/`-layout? Koden importeras som ett riktigt paket (`from iris_service.api import app`)
oavsett var man står. Testerna kör mot det installerade paketet, inte mot filer som råkar
ligga i samma mapp. Det är samma sätt som riktiga bibliotek är byggda på.

## Lägga till beroenden

```bash
uv add requests             # lägger till i pyproject.toml, uppdaterar uv.lock, installerar
uv add --group dev ruff     # utvecklingsberoende, installeras inte i produktion
uv remove requests
```

Skriv aldrig i `uv.lock` för hand. Commita både `pyproject.toml` och `uv.lock`.
Commita aldrig `.venv/`: den är stor, plattformsberoende och återskapas av `uv sync`.

## Vad ska ligga i git?

| Commita | Commita inte |
|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version` | `.venv/` |
| `src/`, `tests/`, `README.md`, `.gitignore` | `__pycache__/`, `.pytest_cache/` |
| `model/` **om** ni väljer att skicka med modellen (se fallgrop 6) | `.DS_Store` och andra OS-filer |

## Vanliga fallgropar

1. **Preprocessing följer inte med modellen.** Spara ett `Pipeline`-objekt, inte bara estimatorn. Annars matas rå, oskalad data in i en modell som tränats på skalad data.
2. **Versionsberoenden.** En joblib-fil är en pickle av Python-objekt. Byter scikit-learn version kan laddningen krascha eller ge tyst fel. Pinna versioner i `pyproject.toml` och commita `uv.lock`; `api.py` varnar om versionen skiljer sig från metadata.
3. **Modellen laddas per request.** Ladda en gång vid uppstart (`lifespan` i FastAPI, `st.cache_resource` i Streamlit).
4. **Feature-ordning.** Kolumnordningen i requesten måste matcha träningsdatan. Se `IrisFeatures.to_array()`.
5. **Hårdkodade absoluta sökvägar.** Alla sökvägar går via `config.py`, som räknar ut projektroten från sin egen plats. Då fungerar repot på en annan dator och oavsett vilken mapp man kör från.
6. **Modellfilen ligger inte i repot.** Antingen commita `model/` (ta bort raden i `.gitignore`) eller se till att `uv run train` kan köras från en ren klon. Projektkrav: "applikationen startar från en ren klon".

## Prova själv

Byt ut Iris mot en modell ni tränade i kapitel 3 eller 4. Vad behöver ändras i `IrisFeatures`? I `Prediction`? Vad händer med `/predict` om er modell är en regressor och inte har `predict_proba`?
