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
persistent deduplication. WhatsApp connection failures are retried once and
logged. Run the app with `uvicorn app.main:app --reload`.