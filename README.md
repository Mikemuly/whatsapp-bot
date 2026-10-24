# whatsapp-bot

Set `VERIFY_TOKEN`, `WHATSAPP_TOKEN`, `PHONE_NUMBER_ID`, `GROQ_API_KEY`, and
`AI_PROVIDER=groq` in a `.env` file (use `.env.example` as a template) to
configure Meta webhook verification and AI-powered replies. Groq uses
`qwen/qwen3.8-27b` by default; set `GROQ_MODEL` to another model enabled for
your account if needed. Set `AI_PROVIDER=gemini` and `GEMINI_API_KEY` to use
Gemini instead. Both provider adapters in `app/ai.py` accept text and return
text; the webhook does not depend on which provider is selected.

The webhook acknowledges messages before processing them in the background and
ignores duplicate message IDs. If generation fails, the selected provider
returns a fallback reply. Message deduplication is in-memory and resets when
the app restarts; production should use Redis or a database for shared,
persistent deduplication. Conversation history is kept in memory per sender and
limited to the latest 10 user/assistant messages; it resets when the app
restarts. WhatsApp connection failures are retried once and logged. Run the app
with `uvicorn app.main:app --reload`.

## Constitution of Kenya answers

The legal-information pipeline uses the official [Constitution of Kenya, 2010
PDF from Kenya Law](https://new.kenyalaw.org/akn/ke/act/2010/constitution/eng@2010-09-03/source),
stored at `data/Constitution_of_Kenya_2010.pdf`. The downloaded copy credits
Kenya Law/Laws.Africa and carries its CC BY-NC-SA 4.0 license. The generated
Chroma index is local at `data/chroma/` and is ignored by Git.

Install the dependencies, then run the one-time ingestion:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe ingest.py
```

Ingestion extracts and cleans PDF text, detects Article headings, splits long
Articles by clause with overlap, prints a text preview and five reproducible
sample chunks, and persists multilingual embeddings to Chroma. Test retrieval
for the housing right:

```powershell
.\venv\Scripts\python.exe rag.py
```

The webhook retrieves up to five relevant excerpts for each question and asks
the selected AI provider to answer only from those excerpts, cite Articles and
clauses, respond in English or Swahili, and include a legal-information
disclaimer.

The Constitution is only one source. Topics such as eviction, employment, and
tenancy may be governed in detail by Acts of Parliament, so the bot should say
when the retrieved Constitution excerpts do not answer a question. Retrieval
can miss relevant text; answers are informational and not legal advice.
