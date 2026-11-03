# whatsapp-bot

This WhatsApp bot is a **Kenyan legal information and case-preparation
assistant**, not a law firm, lawyer, or advocate. It can explain retrieved
legal text and help organize facts, but it does not provide legal advice,
represent users, or predict case outcomes. Users should consult an advocate
for advice about their circumstances.

## Configuration and running

Set `VERIFY_TOKEN`, `WHATSAPP_TOKEN`, `PHONE_NUMBER_ID`, `OPENAI_API_KEY`, and
`AI_PROVIDER=openai` in `.env` (use `.env.example` as a template). OpenAI uses
`gpt-4o-mini` by default; set `OPENAI_MODEL` to another model enabled for your
account if needed. Set `AI_PROVIDER=groq` and `GROQ_API_KEY` to use Groq, or
`AI_PROVIDER=gemini` and `GEMINI_API_KEY` to use Gemini instead.

Run the app with:

```powershell
uvicorn app.main:app --reload
```

The webhook acknowledges messages before processing them in the background
and ignores duplicate message IDs. If generation fails, the selected provider
returns a fallback reply. WhatsApp connection failures are retried once.
Message deduplication is in-memory and resets when the app restarts; production
should use Redis or a database for shared, persistent deduplication.

## Quick question and case preparation modes

- Start with a greeting or `/start`; tap English or Kiswahili to receive a
  short command guide in that language. Tap Other to type a message instead.
  The language preference is held in memory and expires with the conversation.
- Greetings and brief pleasantries receive a short conversational response.
- Send an ordinary message for a brief, cited answer from retrieved legal
  references. If no excerpts are retrieved, the bot says it cannot answer
  reliably rather than filling gaps from model knowledge.
- Send `/case` to begin case preparation. The bot asks short intake questions
  before offering a structured analysis, and it considers prior user-provided
  facts in the retrieval query. Send `/case <brief>` to start with a short
  description.
- Send `/reset` to immediately clear that sender's in-memory conversation and
  leave case mode.
- Send `/summary` to export a short factual summary based only on your
  messages. It is not independently verified and is not legal advice.
- Send `/delete` to clear the same in-memory history and case-mode state.
- Send `/sources` for the indexed documents and their version dates, `/help`
  for commands, or `/start` for the welcome message.

The bot accepts text and interactive menu selections. It replies to voice
notes and images asking the user to type a short description; it does not
transcribe or store those media.

Case mode asks for the timeline, parties, evidence, and desired outcome, one
or two questions at a time. Once the available facts and retrieved provisions
support it, the response covers the core issue, relevant provisions, how they
may apply, strengths and weaknesses, options and forum, deadlines to verify,
and immediate next steps. Urgent safety or liberty concerns take priority.
Criminal matters should be taken promptly to an advocate; the bot must not
coach users to mislead a court.

## Legal references and coverage

The local index uses the following Kenya Law texts. Dates are the versions
shown by Kenya Law on the checked records, not a guarantee that later
amendments do not exist. Re-check the official record before relying on a
provision.

| Source | Indexed PDF | Kenya Law version |
| --- | --- | --- |
| Constitution of Kenya, 2010 | `data/Constitution_of_Kenya_2010.pdf` | [3 September 2010](https://new.kenyalaw.org/akn/ke/act/2010/constitution/eng@2010-09-03) |
| Employment Act | `data/Employment_Act_2007_2024-04-26.pdf` | [26 April 2024](https://kenyalaw.org/akn/ke/act/2007/11/eng@2024-04-26) |
| Criminal Procedure Code | `data/Criminal Procedure Code.pdf` | [11 December 2023](https://kenyalaw.org/akn/ke/act/1930/11/eng@2023-12-11) |
| Penal Code | `data/Penal Code.pdf` | [11 December 2023](https://kenyalaw.org/akn/ke/act/1930/10/eng@2023-12-11) |
| Landlord and Tenant (Shops, Hotels and Catering Establishments) Act | `data/Landlord and Tenant (Shops Hotels and Catering Establishments) Act.pdf` | [31 December 2022](https://kenyalaw.org/akn/ke/act/1965/13/eng@2022-12-31) |
| Rent Restriction Act | `data/Rent Restriction Act.pdf` | [31 December 2022](https://kenyalaw.org/akn/ke/act/1959/35/eng@2022-12-31) |
| Small Claims Court Act | `data/Small Claims Court Act.pdf` | [31 December 2022](https://kenyalaw.org/akn/ke/act/2016/2/eng@2022-12-31) |
| Consumer Protection Act | `data/Consumer Protection Act.pdf` | [31 December 2022](https://kenyalaw.org/akn/ke/act/2012/46/eng@2022-12-31) |
| Law of Succession Act | `data/Law of Succession Act.pdf` | [31 December 2022](https://kenyalaw.org/akn/ke/act/1972/14/eng@2022-12-31) |
| Land Act | `data/Land Act.pdf` | [4 November 2025](https://kenyalaw.org/akn/ke/act/2012/6/eng@2025-11-04) |
| Sexual Offences Act | `data/Sexual Offences Act.pdf` | [26 April 2024](https://kenyalaw.org/akn/ke/act/2006/3/eng@2024-04-26) |
| Data Protection Act | `data/Data Protection Act.pdf` | [31 December 2022](https://kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31) |

Automated ingestion of the statutory texts is covered by the NCLR licensing
permission confirmed by the operator. Kenya Law's source PDFs identify the
NCLR source and provide their reuse/licensing terms; retain attribution and
observe the applicable non-commercial terms. The older
`data/Employment Act.pdf` is not the selected version and is deliberately
excluded.

The generated Chroma index is local at `data/chroma/` and is ignored by Git.
Each chunk records `source`, `section`, and `chapter` metadata (and clause
metadata where available). Citations are rendered as, for example,
“Employment Act, s. 45” or “Constitution of Kenya, 2010, Article 43(1)”.
Answers with missing or unsupported citations are rejected. Retrieval
discards excerpts whose cosine distance is greater than `0.20`, supports an
optional source filter, and expands common Swahili legal terms with English
search terms. If no relevant excerpts remain, the bot says it does not have
the law in its sources rather than improvising.

Install dependencies, then run ingestion:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe ingest.py
```

Ingestion extracts and cleans each PDF, detects Constitution Articles or
statutory sections, splits long provisions by clause with overlap, records
source/section/chapter metadata, prints sample chunks, and persists
multilingual embeddings to Chroma. Test retrieval:

```powershell
.\venv\Scripts\python.exe rag.py
```

Kituo Cha Sheria – Legal Advice Centre identifies itself as a legal-aid
provider on its [official about page](https://kituochasheria.or.ke/about-us/)
(checked 1 November 2026). The prompt suggests checking its official website
for current intake information, but the bot gives no phone numbers or street
addresses. Users should confirm current eligibility and contact details there.
Re-verify other legal-aid organizations and contact information before adding
them to the bot.

## Privacy and retention

The bot stores at most the latest 10 user/assistant messages per WhatsApp
sender in process memory. A cleanup task checks once per minute and deletes
inactive conversations after 24 hours,
clears a conversation immediately on `/reset` or `/delete`, and loses all
conversations on restart. `/delete` removes the bot's in-memory conversation
and case-mode data; it cannot delete messages already sent to WhatsApp or the
configured AI provider. The bot does not write conversation history to disk.
Phone numbers are used to route replies and associate short-lived history;
full message bodies and phone numbers are not intentionally written to
application logs. Messages and relevant history are sent to the configured AI
provider to generate replies, so users should avoid sharing unnecessary
highly sensitive details.
These in-memory controls do not replace a production privacy notice or a
review of Meta and AI-provider data handling.
