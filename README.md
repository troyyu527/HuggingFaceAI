---
title: Troy Yu AI Persona
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12.12"
app_file: app.py
pinned: false
short_description: AI twin of Troy Yu — ask about his BIM and software career
preload_from_hub:
  - BAAI/bge-base-en-v1.5
---

# Troy Yu — AI persona

An AI twin of Troy Yu, a Software Engineer and BIM & CAD Automation Developer in New York. Ask it about his
projects, skills, experience, availability or work authorization, and it answers in his voice from a curated
knowledge base of his career.

**Try it:** https://troy0527-troy-ai-persona.hf.space · Portfolio: https://troyyu527.github.io/ReactWebsite

![The AI persona on its Hugging Face Space: chat box, suggested recruiter questions and the model in use (gemini-3.5-flash)](docs/images/screenshot.png)

> It's an AI assistant: answers come from Troy's knowledge base, but please verify important details with Troy
> (troy0527@gmail.com).

## How it works

```
question ──► search the knowledge base ──► prompt with the best passages ──► LLM ──► streamed answer
             (bge-base embeddings,          (role · summary · rules ·          (Gemini, then Groq
              in memory, top 8)              context)                           if Gemini is busy)
```

- **Knowledge base** — about 80 short passages about Troy's career, written and checked by Troy. They live in a
  **private** Hugging Face Dataset; the app downloads them at startup with a read-only token. They are never part
  of this repository.
- **Search** — every passage is embedded once at startup with `BAAI/bge-base-en-v1.5` (runs on the Space's CPU);
  each question is matched by cosine similarity. A follow-up such as "How big was it?" is first rewritten into a
  complete question using the conversation, so it finds the right project.
- **Answer** — the best passages, Troy's profile summary and a set of rules (first person, career topics only,
  never invent facts, label project values and estimates, respect an NDA, say "I don't have that information"
  when the knowledge base doesn't cover a question) go to the model. The answer streams into the chat.
- **Models** — Google Gemini (`gemini-3.5-flash`, free tier) answers first; when it is unavailable or out of
  quota, the app switches automatically to Groq (`qwen3.8-27b`, then `gpt-oss-120b`, `gpt-oss-20b`, free tier).
  The model that answered is shown at the top of the page.
- **Built with** Python 3.12, Gradio 5, LangChain, sentence-transformers and the OpenAI-compatible APIs of
  Gemini and Groq; hosted on a free Hugging Face Space.

## Privacy and limits

- Questions are sent to Google (Gemini) or Groq to produce the answer. On Gemini's free tier, Google may use
  them to improve its models — please don't type private information.
- The app keeps no conversations. To share the free quota fairly, each visitor (identified by a hashed IP
  address, in memory only) can ask up to **20 questions and about 6,000 tokens a day**.
- The free Space sleeps after a few days without visitors; the first visit then takes about 40 seconds.

## Run your own copy

You need Python 3.12 with [uv](https://docs.astral.sh/uv/), a free Gemini and/or Groq API key, and your own
knowledge base.

1. **Install:** `uv sync`
2. **Settings:** copy `.env.example` to `.env` and fill in the keys. The app reads `LLM_BASE_URL`, `LLM_API_KEY`,
   `LLM_MODEL` (main provider) and optionally `LLM_FALLBACK_BASE_URL`, `LLM_FALLBACK_API_KEY`,
   `LLM_FALLBACK_MODEL`. Any OpenAI-compatible provider works.
3. **Knowledge base:** a `chunks.jsonl` file, one passage per line:

   ```json
   {"id": "profile#00", "doc_id": "profile", "section": "Summary", "text": "…", "metadata": {"title": "…"}}
   ```

   The passage with id `profile#00` is used as the always-present summary. Point `CHUNKS_PATH` at the file.
4. **Run:** `uv run python -m persona` and open http://127.0.0.1:7860.
5. **Deploy to a Hugging Face Space (Gradio SDK):** upload `chunks.jsonl` to a private Dataset
   (`KB_DATASET`, then `uv run python -m persona.upload_kb` with a write token), add the `LLM_*` settings,
   `KB_DATASET` and a **read-only** `HF_TOKEN` as Space secrets, and push this repository to the Space.
