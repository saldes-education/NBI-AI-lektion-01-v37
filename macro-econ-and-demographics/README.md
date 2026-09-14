# Macro economics and demographics

Analysis of how fertility relates to public debt, urbanisation, education and child
mortality in developed countries, plus a "digital twin" state-space model per country
served over HTTP.

The repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with
separate training and serving environments that meet at a versioned artifact directory.

```
data/raw/               source datasets (HFD, JST macrohistory, OECD, OWID, World Bank)
packages/twin_core/     artifact contract shared by training and serving
training/               twin-training: data -> fit -> artifact (CLI: twin-train)
  notebooks/            analysis.ipynb, the exploratory analysis
serving/api/            twin-api: FastAPI service loading one artifact at startup
  postman/              request collection
frontend/               placeholder for the UI
artifacts/baseline/     committed artifact exported by the notebook
artifacts/<version>/    training runs (gitignored)
docker/                 api and train images; compose.yaml wires them together
docs/                   written report
```

## Setup

```bash
make install            # uv sync --all-packages --all-extras (includes notebook deps)
make lint
make test               # make test-fast skips the model-fitting test
```

## Training

```bash
make train              # uv run twin-train  ->  artifacts/<UTC timestamp>/
uv run twin-train --version 2026-09-14 --countries Sweden Japan
make notebook           # analysis; its export cell writes to artifacts/notebook/
```

The notebook export has no manifest, so the API can't load it; use `twin-train` for
anything you want to serve.

An artifact directory holds `manifest.json` (version, training window, library versions,
fit diagnostics), the JSON payload of transition matrices and normalisation, and one pickled
statsmodels results object per country. Directories are never overwritten.

### Backtest

Each country gets a second, evaluation-only fit on 1950–2000 (`--backtest-split`), standardised
with that window's own mean and std, and forecast over 2001–2020. The served model is still the
full 1950–2020 fit. Forecasts are scored by RMSE per variable in real-world units against a
persistence baseline: the last observed 2000 value held flat. The manifest's `diagnostics`
record, per country, `backtest_rmse_<Variable>`, `backtest_baseline_rmse_<Variable>`,
`backtest_skill_<Variable>`, `backtest_train_window_end` and `backtest_n_eval_years`.

Skill is `1 - rmse_model / rmse_baseline`: above 0 the twin beat "nothing changes", 0 ties it,
below 0 it did worse (−1 means twice the error). Compare artifacts on these keys only when they
share the same split. `--no-backtest` skips it; `artifacts/baseline` has no backtest keys.

### Tracking (MLflow)

The artifact directory is the source of truth; MLflow is only a journal for comparing runs, and
the API never reads it.

Install the `tracking` extra (`make install` includes it). With mlflow installed, every
`twin-train` run is recorded unless you pass `--no-tracking`. Without mlflow, training runs as
usual and logs that tracking is off. A tracking error logs a warning and never fails the run.

Runs go to `$MLFLOW_TRACKING_URI`, or a local SQLite file `./mlflow.db` when it is unset
(`--tracking-uri` overrides both), with logged files under `./mlruns/`. No server is needed.
MLflow 3.16 refuses the older `./mlruns` file store unless `MLFLOW_ALLOW_FILE_STORE=true` is
set. Runs are recorded in the `twin-training` experiment (`--experiment`). Each invocation is a
parent run named after `--version`. It holds the CLI settings as parameters, the git commit and
library versions as tags, and the manifest and payload as artifacts (not the pickles). Under it,
one nested run per country holds every numeric diagnostic, including the `backtest_*` scores, as
metrics, with `converged` as a tag.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db    # then open http://127.0.0.1:5000
```

## Serving

```bash
make serve                                            # artifacts/baseline on :8000
TWIN_ARTIFACT_DIR=artifacts/<version> make serve      # serve another artifact
```

| Endpoint          | Purpose                                                        |
| ----------------- | -------------------------------------------------------------- |
| `GET /health`     | status, artifact version, loaded twins                         |
| `GET /countries`  | countries, base year, variables                                |
| `POST /predict`   | forecast in real-world units from the pickled model            |
| `POST /simulate`  | latent-state trajectory from the transition matrix             |
| `GET /`           | interim control panel until `frontend/` exists                 |
| `GET /docs`       | OpenAPI UI                                                     |

`TWIN_CORS_ORIGINS` (comma-separated) enables CORS for a separately hosted frontend.

## Docker

```bash
make docker-train                                     # writes artifacts/<version>/ via a volume
ARTIFACT_DIR=artifacts/<version> docker compose up --build api
```

The API image bakes in exactly one artifact, so a deployed image always serves the same
models. The train image has no notebook or plotting dependencies.

## Known limitations

- **Debt is not per country.** The twins use the cross-country mean of public + private
  debt/GDP from JST (as in the notebook), so every country shares one debt series and
  `debt_to_gdp_pct` is a global indicator. Italy only differs because its window is shorter.
  The backtest therefore scores the same debt series four times over; `backtest_*_Debt` is
  not per-country evidence.
- **Pickles are version-bound.** `statsmodels` is pinned to the same version in training and
  serving; the API logs a warning when an artifact's recorded versions differ from the
  installed ones. Only serve artifacts you produced, since unpickling executes code.
- **No out-of-sample check in the analysis.** Every result in the notebook and report is
  in-sample. The only out-of-sample evaluation is the `twin-train` backtest; in a Sweden run
  the twin did worse than persistence on all five variables.
- **Inputs are interpolated.** Fertility, debt and social spending are filled linearly, and
  urbanisation, schooling and child mortality quadratically, onto an annual grid. Interpolated
  points are not observations, and smoothing inflates significance in the time-series tests.

## Parked work

The per-country SQLite panel for the upcoming ML model (`load_data.py`, `data.db`) is in
`git stash` (`git stash list`, then `git stash pop`).
