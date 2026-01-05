---
trigger: always_on
---

---

alwaysApply: true
---

# Cursor Rules for "Aliyun Call-Center-AI Port"

You are assisting on a long-running project where we are porting / refactoring Microsoft's "call-center-ai" repo from Azure to a China-mainland stack based on Aliyun + Tongyi Qianwen.

## High-level goals

- Maintain the original project architecture and semantics as much as possible.
- Replace Azure-specific services with Aliyun equivalents:
  - Telephony: Azure Communication Services → Aliyun Voice Service (VMS / 智能外呼).
  - STT: Azure Cognitive Services Speech STT → Aliyun NLS 实时语音识别.
  - TTS: Azure Speech TTS → Aliyun NLS TTS or Tongyi TTS (CosyVoice / Sambert / Qwen-TTS).
  - LLM: Azure OpenAI (GPT-4.x) → Tongyi Qianwen (Qwen family via Bailian).
  - Embedding + Search: text-embedding-3-large + Azure AI Search → Tongyi Embedding + Aliyun OpenSearch (向量检索).
  - Data store: Cosmos DB → ApsaraDB (MongoDB preferred; RDS MySQL/PolarDB acceptable).
  - Cache: Azure Redis → ApsaraDB for Redis.
  - Queue/Event: Azure Queue + Event Grid → Aliyun MQ (RocketMQ/MNS) + EventBridge.
  - Container runtime: Azure Container Apps → ACK (Aliyun Kubernetes).
  - Logging/Monitoring: Application Insights → SLS + ARMS/CloudMonitor.
  - Content Safety: Azure Content Safety → Aliyun 内容安全.

## User profile and constraints

- The user is effectively a non-programmer. Assume minimal familiarity with:
  - Coding
  - Cloud providers
  - Docker/Kubernetes
- Always be extremely explicit:
  - Name exact files and paths.
  - Show full code blocks (not "…") for any important logic.
  - Provide exact commands and mention from which directory to run them.
  - Provide sample `.env.example` or `config.aliyun.yaml` entries when introducing new environment variables or configs.

## Process rules

1. **Never do large, sweeping refactors in one step.**
   - Propose a small, coherent change set (e.g., "replace LLM client with Tongyi Qianwen") and complete it end-to-end:
     - Code changes
     - Config changes
     - Local test instructions
   - Then move to the next module.

2. **Prefer diff-style, incremental changes.**
   - When changing existing files, show:
     - The old snippet (if short), and
     - The new snippet with clear explanations.
   - When adding files, specify:
     - Exact path
     - Purpose
     - How it is wired into the rest of the project.

3. **Use Aliyun best practices and official SDKs.**
   - Whenever you introduce code that talks to Aliyun services:
     - Base yourself on the latest official docs / examples.
     - Use environment variables, NOT hardcoded secrets.
     - Mention the approximate SDK version or link you are aligning with.
   - Prefer clear, minimal wrappers over deeply abstracted architectures.

4. **Configuration-first approach.**
   - This project is config-driven; preserve and extend that.
   - When adapting to Aliyun:
     - Introduce Aliyun-specific config files such as `config.aliyun.yaml` OR extend current config with an `aliyun` section.
     - Use clear field names like:
       - `aliyun.vms.access_key_id`
       - `aliyun.vms.region`
       - `aliyun.nls.app_key`
       - `aliyun.llm.api_key`
   - Always update any README or CONFIG docs when adding new config fields.

5. **Local-vs-production separation.**
   - Clearly separate:
     - Local dev mode (no real telephony; mock services or print to console).
     - Production mode (real calls to Aliyun).
   - Provide explicit guidance on:
     - How to run the app locally in "mock telephony" mode.
     - How to configure it for production on ACK.

6. **Testing & verification.**
   - For every non-trivial change:
     - Provide at least one concrete way to verify it:
       - Unit test
       - Integration test
       - Simple script or `curl` call.
   - Explain expected input/output, including dummy payloads.

7. **Kubernetes / ACK deployment.**
   - When we reach deployment phase:
     - Provide minimal, working manifests for ACK:
       - Deployment
       - Service
       - ConfigMap / Secret
     - Use environment variables for Aliyun credentials.
     - Document how to apply them: `kubectl apply -f ...`.

8. **Logging/monitoring.**
   - For any new significant module (telephony, STT, LLM, etc.), ensure:
     - Logs are written in a structured way.
     - Important failures are logged with enough context (without leaking secrets).
   - Later, connect logs to SLS; when doing so, provide any necessary config snippets.

## Style rules

- Be concise but not cryptic. The user needs clarity, not brevity.
- Avoid buzzwords and generic advice; always ground suggestions in concrete code and commands.
- When in doubt between “too detailed” and “too brief”, choose “too detailed”.
- Do not assume pre-existing knowledge of Azure, Aliyun, or Kubernetes internals. When you use them, add one or two sentences explaining what they are doing in this context.

## Safety and secrets

- Never invent or log real keys, secrets, or phone numbers.
- Use obviously fake placeholders like:
  - `ALIYUN_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID`
  - `TEST_PHONE_NUMBER=13800000000`
- Remind the user not to commit `.env` files or real credentials to Git.
## Self-Learning and Evolution

- **Knowledge Persistence**: Follow the global directive in `~/.gemini/GEMINI.md` regarding self-evolution. 
- **Experience Recording**: When encountering global issues or significant optimizations, proactively extract the experience and propose it for addition to the global rules or this file.
