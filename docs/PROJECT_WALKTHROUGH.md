# Call-Center-AI Project Walkthrough (AI Assistant Guide)

This document is a comprehensive, assistant-friendly walkthrough of the project: what it is, how it is structured, how it works at runtime, and what the main functions do.

## 1) Project Overview

**Name**: 品驰关爱中心 AI 客服系统 (call-center-ai)

**Goal**: A low-latency, voice-first AI assistant for postoperative follow-up calls (术后随访) for neuromodulation patients. The system prioritizes fast ASR/LLM/TTS turnaround, SOP-driven data collection, and safety/guardrails for medical contexts.

**Core stack**:
- Python 3.12 + FastAPI
- Aliyun DashScope (Qwen LLM, Paraformer ASR, CosyVoice TTS)
- MongoDB (call state persistence)
- Redis (cache + SOP state persistence)
- FAISS + embeddings for RAG

**Primary user experiences**:
- Streaming voice conversation with VAD and barge-in handling
- Local text/voice chat for testing
- Test platform dashboard for SOP demo data

## 2) High-Level Architecture

### 2.1 Streaming Voice Flow (primary path)

```
Browser mic
  -> /streaming/ UI (PHONE_HTML)
  -> WebSocket /streaming/ws
  -> VAD (client + server)
  -> ASR (Paraformer realtime)
  -> RAG (optional, question-only)
  -> LLM (Qwen)
  -> TTS (CosyVoice streaming)
  -> Audio chunks back to browser
```

Key design points:
- VAD on both client and server sides to minimize latency and avoid unnecessary ASR.
- LLM responses are streamed; TTS synthesizes sentence-by-sentence to start playback early.
- Users can interrupt (barge-in) while TTS is still playing.

### 2.2 Local Chat Flow (legacy, text-first)

```
/local-chat/ UI
  -> POST /local-chat/chat
  -> LLM (Qwen)
  -> optional TTS -> WAV -> browser playback
```

### 2.3 SOP-driven Follow-up Flow

- SOP schema lives in `app/resources/sop_schema.yaml`.
- The SOP engine tracks current step, required slots, retries, and completion status.
- Slot extraction combines regex/keyword heuristics with LLM fallback.

## 3) Repository Structure

```
call-center-ai/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── streaming_routes.py     # WebSocket voice flow + VAD UI
│   ├── local_chat_routes.py    # Local text/voice chat API
│   ├── test_platform.py        # Apple-style demo dashboard
│   ├── helpers/                # Core logic (LLM/ASR/TTS/RAG/SOP/guardrails)
│   ├── models/                 # Pydantic data models
│   ├── persistence/            # Mongo/Redis/memory persistence
│   └── resources/              # SOP schema + RAG data/index
├── public/                     # Static assets (audio, lexicon)
├── docs/                       # Documentation
├── scripts/                    # Dev/diagnostic scripts
├── config.yaml                 # Main runtime configuration
├── docker-compose.yml          # Local stack (app + Mongo + Redis)
└── pyproject.toml              # Dependencies & tooling
```

## 4) Key Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | API root with quick links |
| `/streaming/` | GET | Voice-call UI with VAD |
| `/streaming/ws` | WebSocket | Streaming voice pipeline |
| `/local-chat/` | GET | Local chat UI |
| `/local-chat/chat` | POST | Text chat -> LLM (+ optional TTS) |
| `/local-chat/voice-chat` | POST | Voice upload -> ASR -> LLM -> TTS |
| `/test/` | GET | Demo dashboard with SOP results |
| `/health/readiness` | GET | Readiness checks for cache/store/search |

## 5) Core Modules and Function Meanings

This section lists the most important functions/classes and explains what they do.

### 5.1 API Entrypoints

- `app/main.py`
  - `lifespan()`
    - Pre-warms CosyVoice TTS to reduce first-call latency.
  - `health_readiness_get()`
    - Parallel health checks for Redis, MongoDB, and search backend.
  - `api.include_router(...)`
    - Registers streaming, local chat, and test platform routers.

- `app/streaming_routes.py`
  - `websocket_endpoint()`
    - Main voice call loop: accepts WebSocket, runs VAD, ASR, LLM, TTS, and sends audio/text back to client.
  - `process_audio()`
    - Converts raw audio buffer into ASR text, then calls `process_text()`.
  - `process_text()`
    - Core turn logic: update history, run async slot extraction, run RAG if needed, build system prompt, stream LLM, stream TTS.
  - `recognize_audio()`
    - ASR wrapper around `recognize_audio_sync` with domain hotwords.
  - `stream_llm()`
    - Async generator for streaming Qwen tokens.
  - `synthesize_audio_stream()`
    - Streaming TTS generator that yields PCM chunks.
  - `PHONE_HTML`
    - Voice UI with client-side VAD, audio playback queue, and barge-in control.

- `app/local_chat_routes.py`
  - `chat()`
    - Text -> LLM -> optional TTS -> return JSON.
  - `voice_chat()`
    - WAV/PCM upload -> ASR -> LLM -> optional TTS -> return JSON.

- `app/test_platform.py`
  - `DASHBOARD_HTML`
    - Demo UI to simulate follow-up calls and show Redis-backed SOP results.

### 5.2 LLM, Tools, and Prompting

- `app/helpers/llm_worker.py`
  - `completion_stream()`
    - Streaming LLM with retry and fallback between fast/slow deployments.
  - `completion_sync()`
    - Structured (JSON) completion with validation + retry.
  - `_limit_messages()`
    - Truncates conversation history to fit model context window.

- `app/helpers/llm_tools.py`
  - `DefaultPlugin`
    - Tooling for updating claim data, creating reminders, and knowledge search.

- `app/helpers/llm_utils.py`
  - `AbstractPlugin` + `add_customer_response()`
    - Framework for tool schemas and safe tool execution.

### 5.3 ASR/TTS

- `app/helpers/asr_client.py`
  - `recognize_audio_sync()`
    - Blocking Paraformer recognition with hotwords.

- `app/helpers/tts_client.py`
  - `CosyVoiceSynthesizer`
    - Streaming TTS client with callback-driven audio queue.
  - `synthesize_sync()`
    - One-shot TTS synthesis for local chat.

- `app/helpers/streaming_asr.py`, `app/helpers/persistent_asr.py`
  - Experimental/alternative streaming ASR implementations (persistent session, VAD-driven).

- `app/helpers/streaming_tts.py`
  - Alternative streaming TTS wrapper (simpler, less flexible than full client).

### 5.4 SOP + Slot Extraction

- `app/helpers/sop_engine.py`
  - `SOPEngine.get_current_prompt()`
    - Returns the current SOP question based on state.
  - `SOPEngine.process_response()`
    - Applies extracted slots, validates required fields, advances or retries.

- `app/helpers/slot_extractor.py`
  - `extract_slots()`
    - Hybrid regex + LLM fallback slot extraction (fast path first).

- `app/helpers/history_extractor.py`
  - `extract_from_history()`
    - Periodic LLM extraction from full history (async, fire-and-forget in streaming flow).
  - `get_missing_info()`
    - Returns missing required SOP items for prompt hints.

### 5.5 RAG

- `app/helpers/rag.py`
  - `RAGService.clean_query()`
    - Removes filler words to improve retrieval quality.
  - `RAGService.search()`
    - FAISS vector search with caching and similarity thresholding.

- `app/resources/knowledge_base.json`
  - Source Q&A dataset for RAG.

- `scripts/ingest_qa_data.py`
  - Converts `Q&A_pairs.xlsx` into `knowledge_base.json`.

- `scripts/build_vector_index.py`
  - Builds `faiss_index.bin` + `metadata.pkl` from the knowledge base.

### 5.6 Guardrails + Intent

- `app/helpers/guardrails.py`
  - `check_guardrails()`
    - Rule-based safety checks (e.g., no medication advice, emergency escalation).
  - `should_escalate_to_human()`
    - Escalation logic based on repeated violations.

- `app/helpers/intent_classifier.py`
  - `classify_intent()`
    - Keyword-based intent classification to decide if this is a SOP answer, side question, complaint, etc.

### 5.7 Observability

- `app/helpers/observability.py`
  - `TurnEvent`
    - Structured per-turn snapshot (RAG usage, extraction state, latency, prompt hash).
  - `log_turn_event()`
    - Writes JSONL traces to `traces/` for replay/analysis.

### 5.8 Persistence

- `app/persistence/mongodb.py`
  - `MongoDbStore`
    - CRUD for call state + readiness checks.

- `app/persistence/redis.py`
  - `RedisCache`
    - Cache layer with hashed keys and readiness checks.

- `app/persistence/mock_search.py`
  - Stub search backend for local development.

### 5.9 Local Chat Helpers

- `app/helpers/local_chat.py`
  - `get_or_create_session()`, `get_session()`
    - In-memory session store for the local chat UI.
  - `chat_with_llm()`
    - Builds a short history and calls the fast LLM.
  - `synthesize_speech()`, `recognize_speech()`
    - TTS/ASR helpers for the local chat flow.
  - `pcm_to_wav()`
    - Wraps PCM in a WAV header for browser playback.

### 5.10 Config + Feature Flags

- `app/helpers/config.py`
  - Loads config from `CONFIG_JSON` (env) or `config.yaml`.
  - Validates via Pydantic `RootModel` in `app/helpers/config_models/`.
- `app/helpers/features.py`
  - Simple feature flags and timeouts (VAD timeouts, retry limits).

### 5.11 Data Models

- `app/models/call.py`
  - `CallStateModel` is the main persisted call record.
- `app/models/message.py`
  - `MessageModel`, `PersonaEnum`, `ActionEnum` for conversation history.
- `app/models/readiness.py`, `app/models/training.py`, `app/models/reminder.py`
  - Health checks and auxiliary data types.

## 6) SOP Schema: Structure and Meaning

The SOP schema is defined in `app/resources/sop_schema.yaml`.

Main sections:
- `product_lines`: defines symptom tags by product (e.g., DBS_PD, VNS).
- `flow`: a sequence of steps, each with:
  - `prompt`: what the assistant asks
  - `slots`: structured data to collect
  - `next`: transitions (conditional or unconditional)
- `status_rules`: how to infer final follow-up status
- `guardrails` and `retry`: domain-specific safety and retry rules

This schema is the backbone for the follow-up questionnaire and slot collection.

## 7) Configuration and Environment

### 7.1 `config.yaml`

Main config for runtime defaults (LLM, ASR, TTS, database/cache, conversation settings). Key sections:
- `llm`: Qwen model selections and context sizes
- `asr`: Paraformer model and hotword tuning
- `tts`: CosyVoice model, voice, and sample rate
- `database`, `cache`, `queue`: runtime persistence
- `conversation`, `prompts`: SOP/task prompt templates
- Values are validated and loaded into `CONFIG` at import time.

### 7.2 Environment Variables

Common variables used by the app:
- `DASHSCOPE_API_KEY` (required) - DashScope API access for LLM/ASR/TTS
- `BAILIAN_API_KEY` - optional alias for DashScope key
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`
- `MONGODB_URI`, `MONGODB_DATABASE`
- `CONFIG_JSON` - override config via a JSON blob instead of `config.yaml`
- `PUBLIC_DOMAIN`, `LOG_LEVEL` - runtime display/logging tweaks
- `USE_LANGGRAPH_PLANNER` - enable the LangGraph planner in streaming flow
- `TRACE_ENABLED` - enable/disable JSONL trace logging

## 8) Runtime Behavior (Streaming Call)

1) Client VAD accumulates audio frames, sends chunks when silence is detected.
2) Server VAD does a second pass and feeds ASR.
3) The server:
   - logs user text
   - runs async slot extraction (history-based)
   - uses cached extraction to guide missing-slot hints
   - optionally runs RAG if the user asked a question
   - builds a system prompt that includes SOP guidance + missing slot hints
4) LLM streams tokens; sentences are detected and sent to TTS immediately.
5) TTS streams audio chunks back; UI plays them as they arrive.
6) Turn events are traced for later debugging.

## 9) Latency and Audio Tuning Knobs

Key tuning points for responsiveness and audio quality:
- `app/streaming_routes.py`
  - `ENERGY_THRESHOLD`, `SILENCE_THRESHOLD`, `MIN_SPEECH_DURATION` control server VAD behavior.
  - `TTS_VOICE` and sample rates define TTS voice and playback quality.
- Client-side VAD thresholds live in `PHONE_HTML` (JS constants like `SILENCE_THRESHOLD`, `SILENCE_DURATION`).
- ASR input is 16kHz PCM; streaming TTS outputs 24kHz PCM in the streaming UI.

## 10) Frontend/UI Summary

- `app/streaming_routes.py:PHONE_HTML`
  - Voice-first UI with call-screen metaphor and VAD mic capture.
- `app/local_chat_routes.py` + `app/helpers/local_chat.py`
  - Lightweight text/voice test UI.
- `app/test_platform.py`
  - Dashboard-style UI for demos and SOP result review.

## 11) Scripts and Ops

- `make install` - create venv and install deps
- `docker compose up -d` - start app + MongoDB + Redis
- `make test-static` - Ruff + Pyright
- `make test-unit` - Pytest
- Diagnostics: `scripts/test_llm.py`, `scripts/test_tts.py`, `scripts/test_asr_ws.py`, `scripts/test_mongodb.py`, `scripts/test_redis.py`

## 12) Key Design Principles

- Low latency first: stream as much as possible and minimize blocking calls.
- SOP-driven structure: collect required follow-up data in order with retries.
- Safety: guardrails block unsafe medical advice and escalate emergencies.
- Minimal PII in logs: avoid dumping full transcripts or raw audio.

## 13) Known Gaps / Roadmap Highlights

- Optional LangGraph planner is available but gated behind `USE_LANGGRAPH_PLANNER`.
- OpenSearch-based RAG is noted as future work; current implementation uses FAISS + local JSON.
- Telephony/VMS integration is planned but not yet implemented.

## 14) Quick Pointers for an AI Assistant

If you are assisting development, start here:
- Entry points: `app/main.py`, `app/streaming_routes.py`, `app/local_chat_routes.py`
- SOP logic: `app/resources/sop_schema.yaml`, `app/helpers/sop_engine.py`, `app/helpers/slot_extractor.py`
- RAG: `app/helpers/rag.py`, `app/resources/knowledge_base.json`, `scripts/ingest_qa_data.py`
- Persistence: `app/persistence/mongodb.py`, `app/persistence/redis.py`
- Debugging: `traces/` JSONL turn logs
