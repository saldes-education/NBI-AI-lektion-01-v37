# Frontend

Not started. The API it will talk to lives in `serving/api`:

- OpenAPI schema: `GET /openapi.json` (interactive docs at `/docs`)
- `GET /countries`, `POST /predict`, `POST /simulate`, `GET /health`
- Allow the dev server's origin with `TWIN_CORS_ORIGINS` (compose defaults to `http://localhost:5173`)

Until then, the API serves an interim control panel at `/`
(`serving/api/src/twin_api/static/dashboard.html`).
