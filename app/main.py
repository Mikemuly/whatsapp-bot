import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.responses import PlainTextResponse

from app.ai import (
    LANGUAGE_PICKER_REPLY,
    ConversationMessage,
    Language,
    get_ai_reply,
    is_small_talk_message,
)

logger = logging.getLogger(__name__)

load_dotenv()

HISTORY_TTL_SECONDS = 24 * 60 * 60
HISTORY_CLEANUP_INTERVAL_SECONDS = 60
WHATSAPP_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
WHATSAPP_SEND_ATTEMPTS = 2
WHATSAPP_RETRY_DELAY_SECONDS = 1
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

processed_message_ids: set[str] = set()
MAX_HISTORY_MESSAGES = 10
conversation_history: dict[str, list[ConversationMessage]] = {}
conversation_last_activity: dict[str, float] = {}
case_mode_senders: set[str] = set()
preferred_languages: dict[str, Language] = {}
awaiting_language_other_senders: set[str] = set()

CASE_START_REPLY = (
    "Case preparation mode is on. What happened, and when did it happen? "
    "Start with a brief description; we can go step by step."
)
RESET_REPLY = (
    "Your conversation has been reset. In-memory conversation history and "
    "case mode have been cleared."
)
DELETE_REPLY = (
    "Your in-memory conversation and case-mode data have been deleted. "
    "This cannot remove messages already sent to WhatsApp or the configured "
    "AI provider."
)
WELCOME_REPLY = LANGUAGE_PICKER_REPLY
OTHER_LANGUAGE_REPLY = (
    "Type your message or question below. Legal guidance is currently "
    "available in English or Kiswahili."
)
LANGUAGE_SWITCH_PROMPTS: dict[Language, str] = {
    "english": "Choose your preferred language:",
    "swahili": "Chagua lugha unayopendelea:",
}
UNRECOGNIZED_MESSAGE_REPLIES: dict[Language, str] = {
    "english": (
        "Hi! I'm a Kenyan legal information assistant, not a lawyer. "
        "How may I help you today? Choose a command below, or type /help."
    ),
    "swahili": (
        "Habari! Mimi ni msaidizi wa taarifa za kisheria nchini Kenya, si "
        "wakili. Naweza kukusaidiaje leo? Chagua amri hapa chini, au "
        "andika /help."
    ),
}
LANGUAGE_BUTTON_IDS = {
    "english": "language_english",
    "swahili": "language_swahili",
    "other": "language_other",
}
COMMAND_OPTIONS: dict[Language, list[dict[str, str]]] = {
    "english": [
        {
            "id": "/case",
            "title": "/case - Case prep",
            "description": "Start guided case preparation",
        },
        {
            "id": "/summary",
            "title": "/summary - Summary",
            "description": "Export a factual conversation summary",
        },
        {
            "id": "/sources",
            "title": "/sources - Sources",
            "description": "See indexed laws and versions",
        },
        {
            "id": "/reset",
            "title": "/reset - Reset",
            "description": "Clear conversation and case mode",
        },
        {
            "id": "/delete",
            "title": "/delete - Delete",
            "description": "Delete in-memory conversation data",
        },
        {
            "id": "/help",
            "title": "/help - Commands",
            "description": "Show this command menu",
        },
        {
            "id": "/language",
            "title": "/language - Language",
            "description": "Switch between English and Kiswahili",
        },
        {
            "id": "/start",
            "title": "/start - Welcome",
            "description": "Show welcome and language options",
        },
    ],
    "swahili": [
        {
            "id": "/case",
            "title": "/case - Andaa kesi",
            "description": "Anza kuandaa kesi kwa mwongozo",
        },
        {
            "id": "/summary",
            "title": "/summary - Muhtasari",
            "description": "Pata muhtasari wa ukweli wa mazungumzo",
        },
        {
            "id": "/sources",
            "title": "/sources - Marejeo",
            "description": "Ona sheria na matoleo yaliyoorodheshwa",
        },
        {
            "id": "/reset",
            "title": "/reset - Anza upya",
            "description": "Futa mazungumzo na hali ya kesi",
        },
        {
            "id": "/delete",
            "title": "/delete - Futa data",
            "description": "Futa data ya mazungumzo iliyo kwenye kumbukumbu",
        },
        {
            "id": "/help",
            "title": "/help - Amri",
            "description": "Onyesha menyu hii ya amri",
        },
        {
            "id": "/language",
            "title": "/language - Lugha",
            "description": "Badili kati ya Kiingereza na Kiswahili",
        },
        {
            "id": "/start",
            "title": "/start - Karibu",
            "description": "Onyesha salamu na chaguo la lugha",
        },
    ],
}
HELP_REPLY_ENGLISH = (
    "Commands:\n"
    "/case [facts] - guided case preparation\n"
    "/summary - export a short factual summary\n"
    "/sources - see indexed laws and versions\n"
    "/reset - clear conversation and case mode\n"
    "/delete - delete in-memory conversation data\n"
    "/help - show this message\n\n"
    "/language - switch between English and Kiswahili\n"
    "/start - show the welcome and language choice\n\n"
    "I provide legal information, not legal advice."
)
HELP_REPLY_SWAHILI = (
    "Amri:\n"
    "/case [facts] - kuandaa kesi kwa mwongozo\n"
    "/summary - kupata muhtasari mfupi wa ukweli\n"
    "/sources - kuona sheria zilizoorodheshwa\n"
    "/reset - kufuta mazungumzo na hali ya kesi\n"
    "/delete - kufuta data ya mazungumzo iliyo kwenye kumbukumbu ya muda\n"
    "/help - kuonyesha orodha hii ya amri\n"
    "/language - kubadili kati ya Kiingereza na Kiswahili\n"
    "/start - kuonyesha salamu na chaguo la lugha\n\n"
    "Ninatoa taarifa za kisheria, si ushauri wa kisheria."
)
HELP_REPLIES: dict[Language, str] = {
    "english": HELP_REPLY_ENGLISH,
    "swahili": HELP_REPLY_SWAHILI,
}
HELP_REPLY = HELP_REPLY_ENGLISH
SOURCES_REPLY = (
    "My current indexed sources are:\n"
    "• Constitution of Kenya, 2010 — 3 Sep 2010\n"
    "• Employment Act — 26 Apr 2024\n"
    "• Criminal Procedure Code — 11 Dec 2023\n"
    "• Penal Code — 11 Dec 2023\n"
    "• Landlord and Tenant (Shops, Hotels and Catering Establishments) Act "
    "— 31 Dec 2022\n"
    "• Rent Restriction Act — 31 Dec 2022\n"
    "• Small Claims Court Act — 31 Dec 2022\n"
    "• Consumer Protection Act — 31 Dec 2022\n"
    "• Law of Succession Act — 31 Dec 2022\n"
    "• Land Act — 4 Nov 2025\n"
    "• Sexual Offences Act — 26 Apr 2024\n"
    "• Data Protection Act — 31 Dec 2022\n\n"
    "These are source-version dates, not a guarantee of current law. "
    "Check for later amendments and verify with an advocate."
)
NON_TEXT_REPLY = (
    "I can currently read text messages and menu selections only. Please type "
    "a short description of your question. I don't store or transcribe voice "
    "notes or images. "
    "Kwa sasa ninaweza kusoma ujumbe wa maandishi pekee."
)
SUMMARY_MAX_CHARS = 3000


def _prune_expired_conversations(now: float | None = None) -> None:
    current_time = time.monotonic() if now is None else now
    expired_senders = [
        sender
        for sender, last_activity in conversation_last_activity.items()
        if current_time - last_activity >= HISTORY_TTL_SECONDS
    ]
    for sender in expired_senders:
        conversation_history.pop(sender, None)
        conversation_last_activity.pop(sender, None)
        case_mode_senders.discard(sender)
        preferred_languages.pop(sender, None)
        awaiting_language_other_senders.discard(sender)


def _delete_sender_data(sender: str) -> None:
    conversation_history.pop(sender, None)
    conversation_last_activity.pop(sender, None)
    case_mode_senders.discard(sender)
    preferred_languages.pop(sender, None)
    awaiting_language_other_senders.discard(sender)


def _format_case_summary(
    history: list[ConversationMessage] | None,
) -> str:
    facts = [
        message["content"].strip()
        for message in history or []
        if message["role"] == "user" and message["content"].strip()
    ]
    if not facts:
        return (
            "There are no case facts in this conversation to summarize. "
            "Use /case to begin. This summary does not provide legal advice."
        )

    summary = (
        "*Case preparation summary*\n"
        "Based only on your messages; not independently verified.\n\n"
        "*Facts and timeline as described:*\n"
        + "\n".join(f"{index}. {fact}" for index, fact in enumerate(facts, 1))
        + "\n\n*For an advocate to confirm:* exact dates, documents, witnesses, "
        "and the outcome you want.\n"
        "This is a factual summary, not legal advice."
    )
    if len(summary) > SUMMARY_MAX_CHARS:
        summary = summary[: SUMMARY_MAX_CHARS - 35].rsplit("\n", 1)[0]
        summary += "\n\n[Summary shortened to fit WhatsApp.]"
    return summary


async def _expire_inactive_conversations() -> None:
    while True:
        await asyncio.sleep(HISTORY_CLEANUP_INTERVAL_SECONDS)
        _prune_expired_conversations()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    cleanup_task = asyncio.create_task(_expire_inactive_conversations())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(
    mode: str | None = Query(default=None, alias="hub.mode"),
    verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    challenge: str | None = Query(default=None, alias="hub.challenge"),
) -> PlainTextResponse:
    if (
        mode == "subscribe"
        and VERIFY_TOKEN
        and verify_token == VERIFY_TOKEN
        and challenge is not None
    ):
        return PlainTextResponse(content=challenge, status_code=200)
    return PlainTextResponse(content="Forbidden", status_code=403)


async def send_whatsapp_message(
    to: str,
    message: str,
    language_picker: bool = False,
    supported_language_picker: bool = False,
    command_picker: bool = False,
    language: Language = "english",
) -> None:
    if not WHATSAPP_TOKEN:
        raise RuntimeError("WHATSAPP_TOKEN is not configured")
    if not PHONE_NUMBER_ID:
        raise RuntimeError("PHONE_NUMBER_ID is not configured")

    url = f"https://graph.facebook.com/v23.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    if language_picker or supported_language_picker:
        language_buttons = [
            {
                "type": "reply",
                "reply": {
                    "id": LANGUAGE_BUTTON_IDS["english"],
                    "title": "English",
                },
            },
            {
                "type": "reply",
                "reply": {
                    "id": LANGUAGE_BUTTON_IDS["swahili"],
                    "title": "Kiswahili",
                },
            },
        ]
        if language_picker:
            language_buttons.append(
                {
                    "type": "reply",
                    "reply": {
                        "id": LANGUAGE_BUTTON_IDS["other"],
                        "title": "Other",
                    },
                }
            )
        body = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": message},
                "action": {
                    "buttons": language_buttons
                },
            },
        }
    elif command_picker:
        body = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "body": {"text": message},
                "action": {
                    "button": "Chagua amri"
                    if language == "swahili"
                    else "Commands",
                    "sections": [
                        {
                            "title": "Amri" if language == "swahili" else "Commands",
                            "rows": COMMAND_OPTIONS[language],
                        }
                    ],
                },
            },
        }
    else:
        body = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": message},
        }
    async with httpx.AsyncClient() as client:
        for attempt in range(WHATSAPP_SEND_ATTEMPTS):
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=WHATSAPP_TIMEOUT,
                )
                response.raise_for_status()
                return
            except httpx.HTTPStatusError as error:
                logger.error(
                    "WhatsApp rejected the message with status %s",
                    error.response.status_code,
                )
                return
            except httpx.RequestError:
                if attempt + 1 == WHATSAPP_SEND_ATTEMPTS:
                    logger.exception(
                        "WhatsApp message delivery failed after %s attempts",
                        WHATSAPP_SEND_ATTEMPTS,
                    )
                    return
                logger.warning(
                    "WhatsApp connection failed; retrying in %s second",
                    WHATSAPP_RETRY_DELAY_SECONDS,
                    exc_info=True,
                )
                await asyncio.sleep(WHATSAPP_RETRY_DELAY_SECONDS)


async def process_incoming_message(
    sender: str,
    text: str,
    history: list[ConversationMessage],
    case_mode: bool,
    history_buffer: list[ConversationMessage],
) -> None:
    command_picker = False
    if case_mode and not text:
        reply = CASE_START_REPLY
    elif not case_mode and not is_small_talk_message(text):
        language = preferred_languages.get(sender) or "english"
        reply = UNRECOGNIZED_MESSAGE_REPLIES[language]
        command_picker = True
    else:
        command_picker = False
        language = preferred_languages.get(sender)
        if language:
            reply = await get_ai_reply(
                text,
                history,
                case_mode=case_mode,
                language=language,
            )
        else:
            reply = await get_ai_reply(text, history, case_mode=case_mode)
    if conversation_history.get(sender) is not history_buffer:
        return
    history_buffer.append({"role": "assistant", "content": reply})
    del history_buffer[:-MAX_HISTORY_MESSAGES]
    if reply == WELCOME_REPLY:
        await send_whatsapp_message(sender, reply, language_picker=True)
    elif command_picker:
        await send_whatsapp_message(
            sender,
            reply,
            command_picker=True,
            language=language,
        )
    else:
        await send_whatsapp_message(sender, reply)


@app.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    payload: Any = await request.json()
    entries = payload.get("entry") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        return {"status": "ok"}

    changes = entries[0].get("changes")
    if not isinstance(changes, list) or not changes or not isinstance(changes[0], dict):
        return {"status": "ok"}

    value = changes[0].get("value")
    if not isinstance(value, dict):
        return {"status": "ok"}

    messages = value.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        return {"status": "ok"}

    message = messages[0]
    message_id = message.get("id")
    if not isinstance(message_id, str) or message_id in processed_message_ids:
        return {"status": "ok"}

    sender = message.get("from")
    if not isinstance(sender, str):
        return {"status": "ok"}

    message_type = message.get("type")
    if message_type == "interactive":
        interactive = message.get("interactive")
        button_reply = (
            interactive.get("button_reply")
            if isinstance(interactive, dict)
            else None
        )
        list_reply = (
            interactive.get("list_reply")
            if isinstance(interactive, dict)
            else None
        )
        text_body = (
            button_reply.get("id")
            if isinstance(button_reply, dict)
            else list_reply.get("id")
            if isinstance(list_reply, dict)
            else None
        )
        if not isinstance(text_body, str):
            processed_message_ids.add(message_id)
            background_tasks.add_task(send_whatsapp_message, sender, NON_TEXT_REPLY)
            return {"status": "ok"}
    elif message_type == "text":
        text = message.get("text")
        text_body = text.get("body") if isinstance(text, dict) else None
        if not isinstance(text_body, str):
            return {"status": "ok"}
    else:
        processed_message_ids.add(message_id)
        background_tasks.add_task(send_whatsapp_message, sender, NON_TEXT_REPLY)
        return {"status": "ok"}

    _prune_expired_conversations()
    normalized_text = text_body.strip()
    command = normalized_text.casefold()
    language_choices: dict[str, Language] = {
        "english": "english",
        "en": "english",
        "language_english": "english",
        "swahili": "swahili",
        "kiswahili": "swahili",
        "language_swahili": "swahili",
    }
    if command == LANGUAGE_BUTTON_IDS["other"]:
        awaiting_language_other_senders.add(sender)
        conversation_last_activity[sender] = time.monotonic()
        processed_message_ids.add(message_id)
        background_tasks.add_task(
            send_whatsapp_message,
            sender,
            OTHER_LANGUAGE_REPLY,
            supported_language_picker=True,
        )
        return {"status": "ok"}

    if sender in awaiting_language_other_senders:
        awaiting_language_other_senders.discard(sender)

    if command in language_choices:
        language = language_choices[command]
        preferred_languages[sender] = language
        awaiting_language_other_senders.discard(sender)
        conversation_last_activity[sender] = time.monotonic()
        processed_message_ids.add(message_id)
        background_tasks.add_task(
            send_whatsapp_message,
            sender,
            HELP_REPLIES[language],
            command_picker=True,
            language=language,
        )
        return {"status": "ok"}

    if command == "/language":
        language = preferred_languages.get(sender, "english")
        processed_message_ids.add(message_id)
        background_tasks.add_task(
            send_whatsapp_message,
            sender,
            LANGUAGE_SWITCH_PROMPTS[language],
            supported_language_picker=True,
        )
        return {"status": "ok"}

    if command in {"/reset", "/delete"}:
        _delete_sender_data(sender)
        processed_message_ids.add(message_id)
        reply = DELETE_REPLY if command == "/delete" else RESET_REPLY
        background_tasks.add_task(send_whatsapp_message, sender, reply)
        return {"status": "ok"}

    preferred_language = preferred_languages.get(sender)
    if preferred_language:
        start_reply = (
            "Karibu tena!\n\n" + HELP_REPLIES[preferred_language]
            if preferred_language == "swahili"
            else "Welcome back!\n\n" + HELP_REPLIES[preferred_language]
        )
    else:
        start_reply = WELCOME_REPLY
    direct_replies = {
        "/start": start_reply,
        "/help": HELP_REPLIES.get(preferred_language or "english", HELP_REPLY),
        "/sources": SOURCES_REPLY,
    }
    if command in direct_replies:
        processed_message_ids.add(message_id)
        if command == "/start" and not preferred_language:
            background_tasks.add_task(
                send_whatsapp_message,
                sender,
                direct_replies[command],
                language_picker=True,
            )
        elif command in {"/help", "/start"}:
            background_tasks.add_task(
                send_whatsapp_message,
                sender,
                direct_replies[command],
                command_picker=True,
                language=preferred_language or "english",
            )
        else:
            background_tasks.add_task(
                send_whatsapp_message,
                sender,
                direct_replies[command],
            )
        return {"status": "ok"}

    if command == "/summary":
        history = conversation_history.get(sender, [])
        processed_message_ids.add(message_id)
        background_tasks.add_task(
            send_whatsapp_message,
            sender,
            _format_case_summary(history),
        )
        return {"status": "ok"}

    case_mode = sender in case_mode_senders
    if normalized_text.casefold() == "/case" or normalized_text.casefold().startswith(
        "/case "
    ):
        case_mode = True
        case_mode_senders.add(sender)
        text_body = normalized_text[5:].strip()

    history = conversation_history.setdefault(sender, [])
    conversation_last_activity[sender] = time.monotonic()
    history_snapshot = (
        history[-(MAX_HISTORY_MESSAGES - 1) :]
        if MAX_HISTORY_MESSAGES > 1
        else []
    )
    if text_body:
        history.append({"role": "user", "content": text_body})
        del history[:-MAX_HISTORY_MESSAGES]

    processed_message_ids.add(message_id)
    background_tasks.add_task(
        process_incoming_message,
        sender,
        text_body,
        history_snapshot,
        case_mode,
        history,
    )
    return {"status": "ok"}