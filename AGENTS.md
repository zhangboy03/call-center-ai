# Repository Guidelines

## Project Structure & Module Organization
- FastAPI entrypoints: `app/main.py`, `app/streaming_routes.py` (core streaming voice flow: VAD JS + `/streaming/ws` WebSocket + ASR/LLM/TTS), `app/local_chat_routes.py` (legacy text chat).
- Core helpers: `app/helpers/` (config, logging, LLM/ASR/TTS clients, RAG utilities with query cleaning/cache), `app/models/`, `app/persistence/` (Mongo/Redis/in-memory). Prompts/system glue live in `app/streaming_routes.py`.
- Resources & docs: `app/resources/` (knowledge base JSON), `public/` (static), `docs/`, `HANDOVER.md` (migration + perf notes), `QUICK_REFERENCE.md` (runbook). Runtime config: `config.yaml`; secrets template: `env.example`.
- Ops: `docker-compose.yml` (Mongo/Redis/app), `Makefile`, `cicd/`, `scripts/` (LLM/ASR/TTS smoke tests).

## Build, Test, and Development Commands
- Bootstrap: `cp env.example .env` then set `DASHSCOPE_API_KEY`; install: `make install` (uv venv + deps).
- Run stack: `docker compose up -d`; UI at `http://localhost:8080/streaming/`; logs `docker compose logs -f app`; rebuild `docker compose down && docker compose up -d --build`.
- Dev/hot reload: `make dev` (requires Mongo/Redis). Static checks: `make test-static` (Ruff + Pyright + Bicep). Unit tests: `make test-unit` or `uv run pytest tests/`.
- Smoke inside container: `docker compose exec app python scripts/test_llm.py` / `test_tts.py` / `test_asr_ws.py` / `test_mongodb.py` / `test_redis.py`.

## Coding Style & Naming Conventions
- Python 3.12, 4-space indent; async-first for I/O. Ruff governs style (`ruff check` with isort combine-as-imports); Pyright must be clean; format before PR.
- Naming: modules/files `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`. Keep config fields aligned with `app/helpers/config_models/`.
- Logging: `structlog` with low-latency discipline; keep retries/timeouts in helper layers. Frontend VAD tuning sits in `app/streaming_routes.py:PHONE_HTML`.

## Testing Guidelines
- Unit/integration tests live in `tests/` (`test_*.py`); mock LLM/ASR/TTS and assert `/streaming/ws` payload/turn sequencing. Validate config loaders and missing-env paths for new knobs.
- Scripts under `scripts/` are manual diagnostics, not CI gates. When changing persistence, add tests for Mongo/Redis branches.

## Commit & Pull Request Guidelines
- Conventional Commit prefixes (`feat:`, `fix:`, `refactor:`, `chore:`, `doc:`, `perf:`). Keep PRs scoped; call out endpoint impacts, latency-sensitive changes (VAD/ASR/LLM/TTS), new env vars.
- PR checklist: summary, linked issue/task, affected modules, config changes (no secrets), screenshots for UI, latest test output (`make test-static`, `make test-unit` or relevant script).

## Security, Config, and Latency Tips
- Secrets stay in `.env`; never commit. `config.yaml` must match Pydantic models; common knobs: `llm.tongyi.model`, `tts.cosyvoice.voice`, `asr.paraformer.model`, VAD constants in `streaming_routes.py`.
- Default Compose includes Mongo/Redis; standalone runs must point cache/db URLs to reachable services and verify via `/health/readiness`.
- Mask PII in logs; avoid dumping call audio/text. Prioritize latency: keep prompts short, avoid heavy LLM calls in tight loops, and tune VAD cautiously. RAG already gates on question-like text, strips fillers, and caches results—preserve that when extending.

## Voice Pipeline Best Practices (ASR/LLM/TTS)
- ASR: Prefer persistent Paraformer sessions (16 kHz mono) with server VAD, `language_hints=['zh']`, heartbeat, and hotwords. Send steady 100–200 ms PCM chunks; remove byte-length drops; resample client- or server-side to the model rate.
- Capture/echo: Use AudioWorklet if possible; keep echoCancellation/noiseSuppression on; hard-mute mic while TTS plays and flush pending audio on interrupt to avoid echo loops.
- LLM: Keep prompts short; provide a rolling summary + last few turns; surface pending SOP items in natural language (no slot IDs); use few-shot for friendly + supportive tone; post-check length/end-with-question instead of over-constraining.
- Slot fill: Fast regex/keywords first, LLM fallback only for required slots; normalize to enums/scores before persisting.
- RAG: Run in an executor with low top_k; cache results; answer briefly then return to SOP.
- TTS: Pre-warm CosyVoice; pick one sample rate end-to-end; stream and call `streaming_complete`; allow barge-in to drop remaining audio.

## Roadmap & Current Focus
- Completed: Azure → Aliyun migration (Qwen-turbo, Paraformer, CosyVoice), streaming UI with VAD, 24kHz TTS, ~1–1.5s e2e latency.
- In progress/next: SOP state machine for术后随访 (slot schema, status inference, Redis resume, barge-in handling), RAG upgrades (metadata filters, hybrid search, rewrite fallback, rerank), VMS phone integration, optional streaming ASR, ACK deployment, monitoring/observability. Keep latency-first decisions when adding logic.
