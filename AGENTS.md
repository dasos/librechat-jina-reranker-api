# Agent Guide: Jina Reranker API for LibreChat

## Architecture Notes

- Single-process FastAPI server backed by a separate multiprocessing worker (`model_process`) that holds the model and performs reranking.
- Model loads lazily on first request; `/ready` blocks until status=`ready`.
- All heavy inference happens in the worker process to keep the main loop responsive.
- Cache directory is mounted as `jina_cache` volume for persistent ONNX files between container restarts.

## Commands & Environment

### Local run (Python)
```bash
python main.py
# or, with custom port/host:
SERVER_PORT=9000 SERVER_HOST=127.0.0.1 python main.py
```

### Build Docker image locally
The `Dockerfile` pre-downloads the model during build; disable if you want a smaller layer and first-request loading:
```bash
docker build \
  --build-arg MODEL_NAME=jinaai/jina-reranker-v2-base-multilingual \
  --build-arg CACHE_DIR=/app/.cache \
  -t jina-api .
```

### Docker Compose (recommended)
Shares host cache across restarts. The `jina_cache` volume must persist outside the container:
```bash
docker compose up --build
# or for hot-reload with local source, mount the app directory to /app and skip COPY in CMD:
docker compose -f docker-compose.yml \
  --env-file ./.env.dev \
  build && docker compose down; docker compose up
```

## Endpoints & Status Checks

- `/health` — basic Liveness probe (`{"status":"ok"}`)
- `/ready` — Readiness probe. Raises `503 Service Unavailable` while model is loading or errored, otherwise returns current state with elapsed seconds. Use this in Kubernetes HPA/ingress logic to ensure requests are routed only after the worker has finished starting and reports status=`ready`.
  ```bash
  curl -v http://localhost:8000/ready
  # Wait for "status": "ready" before sending rerank calls if using manual scaling.
  ```

- `/status` — Full runtime state including model name, worker PID, elapsed seconds since `loading`, and last error (if any).

### Reranking endpoint (LibreChat proxy)

```bash
curl -X POST http://localhost:8000/librechat/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "\"foo\"",
    "documents": ["\"bar\"", "\"baz\""],
    "top_n": null,
    "return_documents": false
  }'

```

## Testing & Verification Steps

Because there is no test framework in the repo and model loading depends on external GPU/CPU resources with `fastembed`, follow these steps before introducing changes:

1. Start or restart the server (local Docker) to ensure `/ready` returns success for your modified code path(s).
2. Send a small rerank call (`top_n=2`) to verify worker responsiveness and correct scoring output.
3. Inspect `/status` logs in the application terminal during startup; model loading prints timing information that confirms initialization completed under `CACHE_DIR`.

If you modify batching or timeout parameters, watch for 504 errors from `queue.Empty` on the main thread exceeding `RERANK_TIMEOUT_SECONDS=300s`, which indicates an issue with worker queue handling.

## Environment Variables

| Variable            | Default                          | Purpose                                              |
|---------------------|----------------------------------|------------------------------------------------------|
| `SERVER_HOST`       | `"0.0.0.0"`                     | Bind address for the server                           |
| `SERVER_PORT`       | `8000`                            | Port to listen on                                      |
| `MODEL_NAME`        | `"jinaai/jina-reranker-v2-base-multilingual"` | Model name loaded by fastembed              |
| `CACHE_DIR`         | `/app/.cache` (Docker) or current dir path where main.py is executed    | Path to store the downloaded ONNX model files             |
| `BATCH_SIZE`        | `"32"`                            | Batch size used during inference                      |
| `RERANK_TIMEOUT_SECONDS`  | `"300"`                    | Main-thread timeout waiting for worker queue results   |

## Docker Notes & Gotchas

- The base image uses Python 3.12-slim; keep it minimal to avoid build bloat and CVE expansion.
- Cache directory must be writable by the `app` user (Dockerfile sets permissions); otherwise model loading fails inside containers with a permission error in `/status`.
- If you disable pre-download at runtime, first requests will trigger worker startup latency of 10–20s depending on hardware and network. In Kubernetes deployments that must avoid cold start spikes, ensure `CACHE_DIR` persists across pods or use an InitContainer that verifies model presence before serving traffic.

## Troubleshooting Common Issues

- **503 / Model not ready**: Check `/status`; while it shows status=`loading`, the elapsed seconds increases until status flips to `ready`. A non-zero worker PID confirms a separate process exists; if you see rapid resets (worker dying and restarting), check for GPU OOM or missing ONNX dependencies.
- **504 / Timeout exceeded**: Verify that the queue depth hasn't grown beyond expected limits under load, which can cause main-thread waits to exceed `RERANK_TIMEOUT_SECONDS`. Consider tuning `DEFAULT_BATCH_SIZE` down if you observe request queuing in production traces.

## File Responsibilities

| File               | Purpose                                              |
|--------------------|------------------------------------------------------|
| `api.py`           | Core logic: worker process, endpoint handlers        |
| `func.py`          | Token-counting helpers (`get_rough_token_count`)     |
| `models.py`        | Pydantic request/response schemas for Jina format    |
| `main.py`          | FastAPI app bootstrap + uvicorn launcher             |
