# Från notebook till körbar tjänst – med MLflow som modellregister

> **MLflow är frivilligt i projektet.** Kravet är att modellen har en identitet (version, mätvärde, feature-ordning, sklearn-version) och att ni kan byta version kontrollerat. En `metadata.json` bredvid en joblib-fil med versionsnumrerade filnamn uppfyller det. MLflow är ett erbjudande, inte ett krav – se `01-produktionssättning/` för varianten utan MLflow.

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
uv sync                                                # 0. skapar .venv och installerar allt (inkl. dev-beroenden)

uv run train                                           # 1. tränar, loggar körningen, registrerar version N
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # 2. webbgränssnitt -> http://127.0.0.1:5000
uv run uvicorn iris_mlflow.api:app --reload            # 3. startar API:et -> http://127.0.0.1:8000/docs
uv run pytest -v                                       # 4. kör testerna
uv run streamlit run src/iris_mlflow/ui.py             # 5. alternativt gränssnitt, samma modell
```

`uv sync` laddar själv ner rätt Python-version (se `.python-version`) om den saknas.
Ingen `source .venv/bin/activate` behövs: `uv run` aktiverar miljön åt dig.

`uv run train` fungerar för att `pyproject.toml` deklarerar ett kommando under
`[project.scripts]` som pekar på funktionen `main` i `train.py`. Vill ni ha fler
kommandon (t.ex. `serve`) är det där de läggs till.

Allt MLflow sparar hamnar lokalt: `mlflow.db` (SQLite: körningar, mätvärden, registret)
och `mlruns/` (artefakterna). Inget lämnar datorn. Kommandot för `mlflow ui` körs
från projektroten, eftersom sökvägen `sqlite:///mlflow.db` är relativ.

### Testa API:et manuellt

När API:et är igång: öppna <http://127.0.0.1:8000/docs> och prova `/predict` i Swagger UI,
eller anropa det från terminalen:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Förväntat svar: `{"species": "setosa", "class_index": 0, "probabilities": {...}, "model_version": "1"}`.
Skicka in ett negativt värde eller utelämna ett fält och ni får HTTP 422 tillbaka.

## Struktur

```
02-MLFlow/
├── pyproject.toml          # Paketnamn, beroenden (pinnade!), kommandon
├── uv.lock                 # Exakta versioner av ALLA paket, även indirekta
├── .python-version         # Python-version som uv använder
├── .gitignore              # .venv/, mlflow.db, mlruns/, cache: sådant som inte hör hemma i git
├── src/
│   └── iris_mlflow/        # Själva paketet
│       ├── __init__.py     # Gör mappen till ett importerbart paket
│       ├── config.py       # Sökvägar och namn (tracking-URI, modellnamn, alias), utgår från projektroten
│       ├── train.py        # Steg 1: tränar Pipeline, loggar till MLflow, registrerar version + alias "production"
│       ├── api.py          # Steg 3: FastAPI-app, laddar models:/iris-classifier@production vid uppstart, /health och /predict
│       └── ui.py           # Steg 5: Streamlit-frontend som laddar samma modell från samma register
├── tests/
│   └── test_api.py         # Steg 4: tester med TestClient
├── mlflow.db               # MLflow-registret, SQLite (skapas av train, ligger inte i git)
├── mlruns/                 # Artefakter: modellfiler, MLmodel, signatur (skapas av train, ligger inte i git)
└── .venv/                  # Virtuell miljö (skapas av uv sync, ligger inte i git)
```

Varför `src/`-layout? Koden importeras som ett riktigt paket (`from iris_mlflow.api import app`)
oavsett var man står. Testerna kör mot det installerade paketet, inte mot filer som råkar
ligga i samma mapp. Det är samma sätt som riktiga bibliotek är byggda på.

## Vad MLflow ger oss – artefaktens fyra frågor

| Fråga | Utan MLflow | Med MLflow |
|---|---|---|
| **Vem är jag?** | `metadata.json` skriven för hand | Varje körning loggad; modellen har namn, versionsnummer och alias |
| **Vad behöver jag?** | `sklearn_version` i JSON | `MLmodel` innehåller sklearn-version, `requirements.txt` och Python-version automatiskt |
| **Ser datan rätt ut?** | Kolumnordning hårdkodad i API:et | Signaturen i artefakten anger feature-namn och ordning; API:et läser den |
| **Hur vet någon om jag har fel?** | Sanity check i tester | Samma – plus att varje versions accuracy finns loggad att jämföra mot |

Titta i `mlruns/models/<id>/artifacts/MLmodel` efter en träning. Det är artefaktens pass.

## Alias: så byter man modell i drift utan att röra API:et

```bash
uv run train          # skapar version 2, flyttar aliaset "production" dit
# starta om API:et -> /health visar model_version: 2
```

Blev version 2 sämre? Flytta tillbaka aliaset:

```bash
uv run python -c '
import mlflow
from iris_mlflow.config import TRACKING_URI, MODEL_NAME, ALIAS
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, ALIAS, 1)
'
```

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
| `mlflow.db` och `mlruns/` **om** ni väljer att skicka med registret (se fallgrop 6) | `.DS_Store` och andra OS-filer |

## Vanliga fallgropar

1. **Preprocessing följer inte med modellen.** Spara ett `Pipeline`-objekt. Annars matas oskalad data in i en modell tränad på skalad data – tyst.
2. **Versionsberoenden.** Artefakten återskapas från den installerade scikit-learn-koden. Annan version → krasch, halvfärdigt objekt eller tyst fel. Pinna versioner i `pyproject.toml` och commita `uv.lock`; `api.py` varnar vid avvikelse.
3. **Feature-ordning.** Med MLflow läses ordningen från signaturen. Utan: den enda platsen är `to_array`, och ingenting kontrollerar den.
4. **Modellen laddas per request.** Ladda en gång (`lifespan` i FastAPI, `st.cache_resource` i Streamlit).
5. **Hårdkodade absoluta sökvägar.** Tracking-URI och artefaktmapp går via `config.py`, som räknar ut projektroten från sin egen plats. Då fungerar repot på en annan dator och oavsett vilken mapp man kör från.
6. **Registret saknas på den andra datorn.** `mlflow.db` och `mlruns/` är lokala. Antingen commita dem (ta bort raderna i `.gitignore`) eller se till att `uv run train` kan köras från en ren klon och återskapa registret. Projektkrav: "applikationen startar från en ren klon".

## Prova själv

Byt ut Iris mot en modell ni tränade i kapitel 3 eller 4. Vad behöver ändras i `IrisFeatures`? I `Prediction`? Vad händer med `/predict` om er modell är en regressor utan `predict_proba`? Träna två varianter, jämför dem i `mlflow ui`, och peka aliaset på den bästa.
