from fastapi import Body, FastAPI, HTTPException
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pathlib import Path
from func import get_rough_token_count
from models import JinaRerankerRequest, JinaRerankerResponse
import logging
import multiprocessing
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "jinaai/jina-reranker-v2-base-multilingual")
CACHE_DIR = os.getenv(
    "CACHE_DIR", str(Path(__file__).parent.absolute() / ".cache")
)
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
DEFAULT_TOP_K = int(os.getenv("TOP_K_MAX", "0"))

RERANK_TIMEOUT_SECONDS = int(os.getenv("RERANK_TIMEOUT_SECONDS", "300"))

app = FastAPI()
worker = None
jobs = None
results = None
events = None
rerank_lock = threading.Lock()
state_lock = threading.Lock()
state: dict[str, Any] = {
    "status": "not_started",
    "model": MODEL_NAME,
    "cache_dir": CACHE_DIR,
    "started_at": None,
    "finished_at": None,
    "elapsed_seconds": 0,
    "last_error": None,
    "worker_pid": None,
    "_started_monotonic": None,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def proc_status(pid: int | None = None) -> dict[str, str]:
    proc_path = "/proc/self/status" if pid is None else f"/proc/{pid}/status"
    fields = {}
    wanted = {"State", "VmRSS", "VmSize", "VmHWM", "Threads"}

    try:
        with open(proc_path, encoding="utf-8") as proc_file:
            for line in proc_file:
                key, _, value = line.partition(":")
                if key in wanted:
                    fields[key] = value.strip()
    except OSError:
        pass

    return fields


def set_state(**updates: Any) -> None:
    with state_lock:
        state.update(updates)


def drain_events() -> None:
    if events is None:
        return

    while True:
        try:
            set_state(**events.get_nowait())
        except queue.Empty:
            return


def get_state() -> dict[str, Any]:
    drain_events()

    with state_lock:
        output = dict(state)

    if output["status"] == "loading" and output["_started_monotonic"]:
        output["elapsed_seconds"] = int(time.monotonic() - output["_started_monotonic"])

    output.pop("_started_monotonic", None)
    output["api_process"] = {"pid": os.getpid(), "status": proc_status()}

    worker_pid = output.get("worker_pid")
    if worker_pid:
        output["worker_process"] = {
            "pid": worker_pid,
            "alive": worker.is_alive() if worker else False,
            "status": proc_status(worker_pid),
        }

    return output


def log_startup_files() -> None:
    cache_path = Path(CACHE_DIR)
    logger.info(
        "Cache dir: path=%s exists=%s readable=%s writable=%s",
        cache_path,
        cache_path.is_dir(),
        os.access(cache_path, os.R_OK),
        os.access(cache_path, os.W_OK),
    )

    model_cache = cache_path / "models--jinaai--jina-reranker-v2-base-multilingual"
    onnx_files = []
    for path in sorted(model_cache.glob("snapshots/*/onnx/*")):
        if not path.is_file():
            continue

        try:
            resolved = path.resolve()
            onnx_files.append(
                {
                    "path": str(path.relative_to(cache_path)),
                    "resolved": str(resolved),
                    "size": resolved.stat().st_size,
                }
            )
        except OSError as exc:
            onnx_files.append({"path": str(path), "error": str(exc)})

    logger.info("Cached ONNX files: %s", onnx_files)


def document_text(document: str | dict) -> str:
    if isinstance(document, str):
        return document

    for key in ("text", "content", "page_content"):
        value = document.get(key)
        if isinstance(value, str):
            return value

    return str(document)


def model_process(job_queue, result_queue, event_queue) -> None:
    started = time.monotonic()
    event_queue.put(
        {
            "status": "loading",
            "started_at": utc_now(),
            "finished_at": None,
            "elapsed_seconds": 0,
            "last_error": None,
            "worker_pid": os.getpid(),
            "_started_monotonic": started,
        }
    )

    try:
        log_startup_files()
        logger.info("Loading reranker model %s from %s", MODEL_NAME, CACHE_DIR)
        encoder = TextCrossEncoder(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
        elapsed = int(time.monotonic() - started)
        event_queue.put(
            {
                "status": "ready",
                "finished_at": utc_now(),
                "elapsed_seconds": elapsed,
                "last_error": None,
            }
        )
        logger.info("Loaded reranker model %s in %ss", MODEL_NAME, elapsed)
    except Exception as exc:
        elapsed = int(time.monotonic() - started)
        event_queue.put(
            {
                "status": "error",
                "finished_at": utc_now(),
                "elapsed_seconds": elapsed,
                "last_error": str(exc),
            }
        )
        logger.exception("Error loading reranker model %s", MODEL_NAME)
        return

    while True:
        job = job_queue.get()
        if job is None:
            return

        request_id = job["request_id"]
        try:
            scores = encoder.rerank(
                job["query"],
                job["documents"],
                batch_size=job["batch_size"],
            )
            result_queue.put(
                {
                    "request_id": request_id,
                    "ok": True,
                    "scores": [float(score) for score in scores],
                }
            )
        except Exception as exc:
            logger.exception("Rerank failed in worker")
            result_queue.put({"request_id": request_id, "ok": False, "error": str(exc)})


@app.on_event("startup")
def start_worker() -> None:
    global worker, jobs, results, events

    context = multiprocessing.get_context("fork")
    jobs = context.Queue()
    results = context.Queue()
    events = context.Queue()
    worker = context.Process(target=model_process, args=(jobs, results, events), daemon=True)
    worker.start()

    set_state(
        status="loading",
        started_at=utc_now(),
        finished_at=None,
        elapsed_seconds=0,
        last_error=None,
        worker_pid=worker.pid,
        _started_monotonic=time.monotonic(),
    )
    logger.info("Started model worker pid=%s", worker.pid)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    current = get_state()
    if current["status"] != "ready":
        raise HTTPException(status_code=503, detail=current)
    return current


@app.get("/status")
def status():
    return get_state()


@app.post("/librechat/v1/rerank", response_model=JinaRerankerResponse)
def rerank(request: JinaRerankerRequest = Body(...)):
    current = get_state()
    if current["status"] != "ready":
        raise HTTPException(status_code=503, detail=current)

    request_id = str(uuid.uuid4())

    try:
        started = time.monotonic()
        with rerank_lock:
            documents = [document_text(document) for document in request.documents]
            batch_size = request.batch_size or DEFAULT_BATCH_SIZE

            jobs.put(
                {
                    "request_id": request_id,
                    "query": request.query,
                    "documents": documents,
                    "batch_size": batch_size,
                }
            )

            try:
                result = results.get(timeout=RERANK_TIMEOUT_SECONDS)
            except queue.Empty:
                raise HTTPException(status_code=504, detail="Rerank request timed out")

            if result["request_id"] != request_id:
                raise RuntimeError("Received rerank result for an unexpected request")

            if not result["ok"]:
                raise HTTPException(status_code=500, detail=result["error"])

        logger.info(
            "Reranked %s documents in %.2fs",
            len(request.documents),
            time.monotonic() - started,
        )

        ranked_results = sorted(
            enumerate(result["scores"]),
            key=lambda item: item[1],
            reverse=True,
        )
        
        effective_top_n = (request.top_n or DEFAULT_TOP_K) if request.documents else 0
        
        rank_len = len(ranked_results)
        limit_idx = min(effective_top_n, rank_len - 1) if rank_len > 0 and effective_top_n >= 0 else None

        if effective_top_n is not None:
            ranked_results = ranked_results[:effective_top_n]

        return {
            "model": MODEL_NAME,
            "usage": {"total_tokens": get_rough_token_count(request.query, documents)},
            "results": [
                {
                    "index": index,
                    "relevance_score": score,
                    "document": request.documents[index] if request.return_documents else None,
                }
                for index, score in ranked_results
            ],
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error handling rerank request")
        raise HTTPException(status_code=500, detail="Error handling request")
